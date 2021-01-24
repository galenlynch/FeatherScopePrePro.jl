module FeatherScopePrePro

using CaProcessing: clip_segments_thr, clip_imgs, demin, map_to_8bit, PixelLUT,
    pixel_lut, rescale_compress, max_container_depth, frame_avg_intensity,
    get_norm, frames_min_max_accum_alloc, frames_min_max_accum_init!,
    frame_min_max_accum!, determine_container_depth, make_scale_f,
    make_pixel_lut, apply_lut!
using GLUtilities: ndx_to_t, t_to_ndx, clip_ndx, find_all_edge_triggers
using ColorTypes: Gray, RGB, RGB24
using DataStructures: OrderedDict, CircularBuffer, isfull

using FeatherscopeExtraction: convert_feather_video_frames,
    convert_featherscope_rgb, open_audio_sync, FEATHER_VIDEO_REG,
    FEATHER_SYNC_REG, sync_searchreg, video_sync_alignment, frame_iter_preamble

using FileIO: save
using FixedPointNumbers: Normed, N6f10, N0f8, N8f8, rawtype # need reinterpret method
using ImageCore: colorview
using JSON: json
using WAV: wavwrite, WAVE_FORMAT_PCM

using Base.Threads: @spawn, nthreads
using Base.Iterators: peel

using Statistics: mean, median

using ImageOverlays: MutableImage, get_image, grid_lines

using Dates: @dateformat_str, DateTime, Millisecond

using VideoIO: open_video_out, VideoWriter, openvideo, append_encode_mux!,
    close_video_out!
import VideoIO

import FFMPEG

export avi_to_scaled_gray_video,
    avis_to_tiff_demin,
    avi_to_tiff_demin,
    avi_to_tiff_raw,
    find_exposed_frame_ranges,
    file_triplets,
    feather_video_encode_demind_segments,
    feather_video_min_frame_planning,
    feather_video_read_demin_audio,
    feather_video_read_demean_grid_write,
    measure_framerate,
    streaming_intensities,
    sync_triplets,
    triplet_sync_info,
    video_sync_alignment

include("sync.jl")

const AVI_REGEX = r"(?<file_prefix>.*)\.avi$"i
const JSON_DICT_TYPE = OrderedDict{String, Any}

function force_file_check(filename, force)
    if !force && isfile(filename)
        error("File $filename already exists, set force = true to overwrite")
    end
end

function finish_exposure!(min_frames, subtracted_maxvals, exposed_ranges,
                          roi_max, roi_min, exposure_start, exposure_stop, nt)
    subtr_mv = maxval_min_max_frames(roi_min, roi_max, nt)
    push!(min_frames, copy(roi_min))
    push!(subtracted_maxvals, subtr_mv)
    push!(exposed_ranges, exposure_start:exposure_stop)
end

function _add_to_min_max_frames!(roi_max, roi_min, img_raw, roi_xr, roi_yr, yb,
                                 ye)
    @inbounds for yi in yb:ye
        @simd ivdep for xi in eachindex(roi_xr)
            intensity = reinterpret(N6f10,
                                    convert_featherscope_rgb(img_raw[roi_xr[xi],
                                                                     roi_yr[yi]]))
            roi_max[xi, yi] = max(intensity, roi_max[xi, yi])
            roi_min[xi, yi] = min(intensity, roi_min[xi, yi])
        end
    end
end

function add_to_min_max_frames!(roi_max, roi_min, img_raw, roi_xr, roi_yr, nt)
    ny = length(roi_yr)
    if nt > 1
        blksize = cld(ny, nt)
        tasks = Vector{Task}(undef, nt)
        @inbounds for tno in 1:nt
            lo = (tno - 1) * blksize + 1
            hi = min(tno * blksize, ny)
            tasks[tno] = @spawn _add_to_min_max_frames!(roi_max, roi_min,
                                                        img_raw, roi_xr,
                                                        roi_yr, lo, hi)
        end
        foreach(wait, tasks)
    else
        _add_to_min_max_frames!(roi_max, roi_min, img_raw, roi_xr, roi_yr, 1,
                                 ny)
    end
end

function exposure_incr!(roi_max, roi_min, min_frames, subtracted_maxvals,
                        exposed_ranges, exposure_start, img_raw,
                        in_exposure, fno, roi_xr, roi_yr, nt)
    if exposure_start > 0 # in exposed period
        if in_exposure # continue exposure
            add_to_min_max_frames!(roi_max, roi_min, img_raw, roi_xr, roi_yr, nt)
        else # end exposure
            finish_exposure!(min_frames, subtracted_maxvals, exposed_ranges,
                             roi_max, roi_min, exposure_start, fno - 1, nt)
            exposure_start = 0
        end
    else # not in exposure period
        if in_exposure # start exposure
            fill!(roi_max, typemin(N6f10))
            fill!(roi_min, typemax(N6f10))
            add_to_min_max_frames!(roi_max, roi_min, img_raw, roi_xr, roi_yr, nt)
            exposure_start = fno
        end # do nothing if still not exposed
    end
    exposure_start
end

function check_slices(img, roi_x, roi_y)
    checkbounds(img, roi_x, roi_y)
    to_indices(img, (roi_x, roi_y))
end

function feather_video_min_frame_planning(input_fname, thr, roi_x = :,
                                          roi_y = :; shutter_delay = 1,
                                          nt = nthreads())
    outs = frame_iter_preamble(input_fname)
    outs === nothing && return
    inputvid, img_raw  = outs
    roi_xr, roi_yr = check_slices(img_raw, roi_x, roi_y)
    norm = get_norm(roi_xr, roi_yr)
    roi_max = Matrix{N6f10}(undef, length(roi_xr), length(roi_yr))
    roi_min = similar(roi_max)

    min_frames = Vector{typeof(roi_min)}()
    subtracted_maxvals = Vector{N6f10}()
    exposed_ranges = Vector{UnitRange{Int}}()

    fno = 1
    exposure_start = 0
    is_exposed = determine_if_frame_exposed(img_raw, roi_xr, roi_yr,
                                            norm, thr, nt)
    nexposed = ifelse(is_exposed, 1, 0)
    in_exposure = nexposed > shutter_delay
    exposure_start = exposure_incr!(roi_max, roi_min, min_frames,
                                    subtracted_maxvals, exposed_ranges,
                                    exposure_start, img_raw,
                                    in_exposure, fno, roi_xr,
                                    roi_yr, nt)
    while !eof(inputvid)
        read!(inputvid, img_raw)
        fno += 1
        is_exposed = determine_if_frame_exposed(img_raw, roi_xr, roi_yr,
                                                norm, thr, nt)
        nexposed = ifelse(is_exposed, nexposed + 1, 0)
        in_exposure = nexposed > shutter_delay
        exposure_start = exposure_incr!(roi_max, roi_min, min_frames,
                                        subtracted_maxvals,
                                        exposed_ranges, exposure_start,
                                        img_raw, in_exposure, fno,
                                        roi_xr, roi_yr, nt)
    end
    if exposure_start > 0 # last frame was exposed, finish exposure
        finish_exposure!(min_frames, subtracted_maxvals, exposed_ranges,
                         roi_max, roi_min, exposure_start, fno, nt)
    end
    return min_frames, subtracted_maxvals, exposed_ranges, fno
end

function avis_to_tiff_demin(savedir, fnames, x, y, thr;
                      scratch_dir = tempdir(), nt = nthreads())
    for fname in fnames
        avi_to_tiff_demin(savedir, fname, x, y, thr, scratch_dir = scratch_dir)
    end
end

function avi_to_tiff_demin(savedir::AbstractString, in_filename::AbstractString,
                           x::AbstractRange, y::AbstractRange, thr::Real;
                           scratch_dir = tempdir(), kwargs...)
    imgs = convert_feather_video_frames(in_filename, scratch_dir = scratch_dir)
    avi_to_tiff_demin(savedir, imgs, in_filename, x, y, thr; kwargs...)
end

function avi_to_tiff_demin(savedir::AbstractString, imgs::AbstractArray,
                           in_filename::AbstractString, x::AbstractRange,
                           y::AbstractRange, thr::Real;
                           nt = nthreads(),
                           name_f = default_name_conversion)
    # convert video and find exposed segments
    imgs_roi = clip_imgs(imgs, x = x, y = y)
    segs, open_pers = clip_segments_thr(imgs_roi, thr, nt = nt)

    nseg = length(segs)
    nframes = [open_pers[2, i] - open_pers[1, i] + 1 for i in 1:nseg]
    keep_idxs = findall(x -> x > 1, nframes)
    nkeep = length(keep_idxs)

    # build metadata json
    pref = extract_avi_prefix(in_filename)
    json_fname = joinpath(savedir, pref * ".json")

    json_dict = JSON_DICT_TYPE()
    json_dict["working_dir"] = pwd()
    json_dict["input_avi"] = in_filename
    json_dict["threshold"] = thr

    xb, xe = x_to_bounds(imgs, x)
    json_dict["x_start"] = xb
    json_dict["x_stop"] = xe

    yb, ye = y_to_bounds(imgs, y)
    json_dict["y_start"] = yb
    json_dict["y_stop"] = ye

# create storage for each output's metadata
    seg_dicts = Vector{JSON_DICT_TYPE}(undef, nkeep)
    json_dict["outputs"] = seg_dicts

    for (keepno, segno) in enumerate(keep_idxs)
        demind = demin(segs[segno], nt = nt)
        if reinterpret(maximum(demind)) < typemax(UInt8)
            output = map_to_8bit(demind)
            depth = 8
        else
            output = demind
            depth = 16
        end

        # save output
        new_bname = name_f(pref, open_pers[1, segno], open_pers[2, segno])
        tiff_fname = joinpath(savedir, new_bname * ".tiff")

        save(tiff_fname, colorview(Gray, PermutedDimsArray(output, (2, 1, 3))))

        # output metadata
        seg_dicts[keepno] = JSON_DICT_TYPE()
        seg_dicts[keepno]["output_name"] = tiff_fname
        seg_dicts[keepno]["frame_start"] = open_pers[1, segno]
        seg_dicts[keepno]["frame_stop"] = open_pers[2, segno]
        seg_dicts[keepno]["bit_depth"] = depth
    end

    # save metadata
    open(json_fname, "w") do io
        print(io, json(json_dict))
    end
end

function  avi_to_tiff_raw(savedir, in_filename; scratch_dir = tempdir())
    pref = extract_avi_prefix(in_filename)
    imgs = convert_feather_video_frames(in_filename, scratch_dir = scratch_dir)
    out_fname = joinpath(savedir, pref * ".tiff")
    save(out_fname, colorview(Gray, PermutedDimsArray(imgs, (2, 1, 3))))
end

x_to_bounds(imgs, ::Colon) = (1, size(imgs, 1))
x_to_bounds(::Any, r::UnitRange) = (first(r), last(r))
y_to_bounds(imgs, ::Colon) = (1, size(imgs, 2))
y_to_bounds(::Any, r::UnitRange) = (first(r), last(r))

function extract_avi_prefix(orig)
    bname = basename(orig)
    m = match(AVI_REGEX, bname)
    m === nothing && throw(ArgumentError("Could not parse filename $orig"))
    m[:file_prefix]
end

function default_name_conversion(pref, b, e)
    join([pref, "frames", string(b), string(e)], '-')
end

function demin_frame!(graybuf, img_raw, min_frame, lut, roi_xr, roi_yr)
    @inbounds for y in eachindex(roi_yr)
        @simd ivdep for x in eachindex(roi_xr)
            intensity = convert_featherscope_rgb(img_raw[roi_xr[x], roi_yr[y]])
            demind = intensity - reinterpret(min_frame[x, y])
            graybuf[x, y] = lut[demind]
        end
    end
end

struct FrameEncoderState{T}
    lut::PixelLUT{T}
    io::IOStream
    s_fpath::String
    writer::VideoWriter
end

function start_encode(out_filename, graybuf, framerate, container_settings,
                      container_private_settings, encoder_settings,
                      encoder_private_settings, force)
    writebuf = PermutedDimsArray(graybuf, (2,1))
    force_file_check(out_filename, force)
    writer = open_video_out(out_filename, writebuf; framerate,
                             container_settings, container_private_settings,
                             encoder_settings, encoder_private_settings)
    return io, s_fpath, encoder
end

function append_frame!(encoder_state, graybuf, fno)
    writebuf = PermutedDimsArray(graybuf, (2,1))
    append_encode_mux!(encoder_state.writer, writebuf, fno)
end

finish_encode(encoder_state) = close_video_out!(encoder_state.writer)

function append_demind_video_frame!(encoder_state, exposure_no, graybuf, img_raw,
                                    exposed_range, min_frame, sub_maxv, fno,
                                    roi_xr, roi_yr, out_filename, framerate,
                                    use_gamma_compression, container_settings,
                                    container_private_settings, encoder_settings,
                                    encoder_private_settings, force)
    if encoder_state === nothing
        if fno in exposed_range
            io, s_fpath, encoder = start_encode(out_filename, graybuf, framerate,
                                                container_settings,
                                                container_private_settings,
                                                encoder_settings,
                                                encoder_private_settings, force)
            maxv_scale = 1 / sub_maxv
            Tout = eltype(graybuf)
            if use_gamma_compression
                lut = pixel_lut(sub_maxv) do x
                    rescale_compress(Tout, x, maxv_scale)
                end
            else
                lut = pixel_lut(x -> convert(Tout, maxv_scale * x), sub_maxv)
            end
            encoder_state = FrameEncoderState(lut, io, s_fpath, encoder)
            demin_frame!(graybuf, img_raw, min_frame, encoder_state.lut, roi_xr,
                         roi_yr)
            append_frame!(encoder_state, graybuf, fno)
        end
    else
        if fno in exposed_range # continue encode
            demin_frame!(graybuf, img_raw, min_frame, encoder_state.lut, roi_xr,
                         roi_yr)
            append_frame!(encoder_state, graybuf, fno)
        else # finish encode
            finish_encode(encoder_state, exposed_range, framerate)
            encoder_state = nothing
            exposure_no += 1
        end
    end
    encoder_state, exposure_no
end

function def_fname_f(bname, r)
    bname_noext, bname_ext = splitext(bname)
    "$(bname_noext)_$(r[1])-$(r[end]).mp4"
end

function feather_video_encode_demind_segments(input_fname, roi_x, roi_y,
                                              min_frames, subtracted_maxvals,
                                              exposed_ranges, framerate,
                                              writedir = pwd();
                                              use_gamma_compression = true,
                                              container_settings = (;),
                                              container_private_settings =
                                              (movflags = "+write_colr",),
                                              encoder_settings = (color_range = 2,),
                                              encoder_private_settings =
                                              (crf = 20, preset = "medium"))
    nexposure = length(exposed_ranges)
    nexposure > 0 || return String[]
    isdir(writedir) || throw(ArgumentError("Cannot access write directory $writedir"))
    base_name = basename(input_fname)
    video_names = joinpath(writedir, def_fname_f.(base_name, exposed_ranges))

    outs = frame_iter_preamble(input_fname)
    outs === nothing && return String[]
    inputvid, img_raw = outs
    roi_xr, roi_yr = check_slices(img_raw, roi_x, roi_y)
    graytype = max_container_depth(subtracted_maxvals)
    graybuf = similar(img_raw, graytype, length.((roi_xr, roi_yr)))

    encoder_state = nothing
    fno = 1
    exposure_no = 1
    encoder_state, exposure_no = append_demind_video_frame!(
        encoder_state, exposure_no, graybuf, img_raw,
        exposed_ranges[exposure_no], min_frames[exposure_no],
        subtracted_maxvals[exposure_no], fno, roi_xr, roi_yr,
        video_names[exposure_no], framerate, use_gamma_compression,
        container_settings, container_private_settings,
        encoder_settings, encoder_private_settings
    )

    while !eof(inputvid) && exposure_no <= nexposure
        VideoIO.read!(inputvid, img_raw)
        fno += 1
        encoder_state, exposure_no = append_demind_video_frame!(
            encoder_state, exposure_no, graybuf, img_raw,
            exposed_ranges[exposure_no], min_frames[exposure_no],
            subtracted_maxvals[exposure_no], fno, roi_xr, roi_yr,
            video_names[exposure_no], framerate,
            use_gamma_compression, container_settings,
            container_private_settings,
            encoder_settings, encoder_private_settings
        )
    end

    if encoder_state !== nothing
        finish_encode(encoder_state, video_names[exposure_no],
                      exposed_ranges[nexposure], framerate)
    end

    return joinpath.(writedir, video_names)
end

function feather_video_read_demin(input_fname, thr, roi_x, roi_y, framerate,
                                  writedir = pwd();
                                  container_settings = (;),
                                  container_private_settings =
                                  (movflags = "+write_colr",),
                                  encoder_settings = (color_range = 2,),
                                  encoder_private_settings =
                                  (crf = 20, preset = "medium"),
                                  use_gamma_compression = true)
    min_frames, subtracted_maxvals, exposed_ranges, nf =
        feather_video_min_frame_planning(input_fname, thr, roi_x, roi_y)
    feather_video_encode_demind_segments(input_fname, roi_x, roi_y, min_frames,
                                         subtracted_maxvals, exposed_ranges,
                                         framerate, writedir;
                                         use_gamma_compression,
                                         container_settings,
                                         container_private_settings,
                                         encoder_settings,
                                         encoder_private_settings)
end

function center_scale(a, max_dev = 1)
    centered = a .- mean(a)
    scale = max_dev / maximum(abs.(extrema(centered)))
    centered .*= scale
end

function feather_sync_add_audio(syncf, out_filenames, exposed_ranges, sync_frameno,
                                framerate, writedir, shutter_offset, fs_sync,
                                force_video, nexposed; audio_depth = 16, kwargs...)
    # Find sync edges
    syncdata = open_audio_sync(syncf)
    sync_start_time = video_sync_alignment(syncdata, sync_frameno, framerate;
                                           kwargs...)

    # Loop over the separated video files and add the overlapping audio
    syncl = size(syncdata, 2)
    temp_video_f = joinpath(writedir, "temp.mp4")
    temp_audio_f = joinpath(writedir, "temp.wav")
    for exposure_no in 1:nexposed
        time_start = ndx_to_t(first(exposed_ranges[exposure_no]), framerate)
        time_stop = ndx_to_t(last(exposed_ranges[exposure_no]) + 1, framerate)
        exposure_sync_start_ndx = clip_ndx(t_to_ndx(time_start, fs_sync,
                                                    sync_start_time), syncl)
        exposure_sync_stop_ndx = clip_ndx(t_to_ndx(time_stop, fs_sync,
                                                   sync_start_time), syncl)
        audio_segment = view(syncdata, 1,
                             exposure_sync_start_ndx : exposure_sync_stop_ndx)
        scaled_audio = center_scale(audio_segment)
        # Make audio file
        wavwrite(scaled_audio, temp_audio_f, Fs = fs_sync, nbits = audio_depth,
                 compression = WAVE_FORMAT_PCM)
        try
            mv(out_filenames[exposure_no], temp_video_f, force = force_video)
            try
                # join video and audio
                FFMPEG.exe(`-y -i $(temp_video_f) -i $(temp_audio_f) -c:v copy
                            -c:a aac -map 0:v:0 -map 1:a:0
                            $(out_filenames[exposure_no])`)
            finally
                rm(temp_video_f)
            end
        finally
            rm(temp_audio_f)
        end
    end
end

function feather_video_read_demin_audio(videof::AbstractString,
                                        syncf::AbstractString, thr::Real, roi_x,
                                        roi_y, framerate, writedir = pwd();
                                        shutter_offset = 1, fs_sync = 48000,
                                        force_video = false,
                                        use_gamma_compression = true,
                                        encoder_properties = (;),
                                        encoder_private_properties = (;),
                                        kwargs...)
    # Find exposed portions of the video and make subtracted videos
    min_frames, subtracted_maxvals, exposed_ranges, nf =
        feather_video_min_frame_planning(videof, thr, roi_x, roi_y)

    # If the only exposed region is at the start of the video file, then we
    # cannot align the audio and should give up
    nexposed = length(exposed_ranges)
    nexposed == 0 && return String[]
    first_exposure_nosync = first(exposed_ranges[1]) == 1
    first_exposure_nosync && nexposed == 1 && return String[]

    out_filenames = feather_video_encode_demind_segments(videof, roi_x, roi_y,
                                                      min_frames,
                                                      subtracted_maxvals,
                                                      exposed_ranges, framerate,
                                                      writedir;
                                                      use_gamma_compression,
                                                      encoder_properties,
                                                      encoder_private_properties)

    sync_exposed_frameno = first_exposure_nosync ? exposed_ranges[2][1] :
                                                   exposed_ranges[1][1]
    try
        feather_sync_add_audio(syncf, out_filenames, exposed_ranges,
                               sync_exposed_frameno, framerate, writedir,
                               shutter_offset, fs_sync, force_video, nexposed;
                               kwargs...)
    catch
        for f in out_filenames
            isfile(f) && rm(f)
        end
        rethrow()
    end
end

function dither_noise(scale::Number = 1)
    samp = rand() + rand()
    samp * scale / 2
end

function feather_video_read_demin_audio(videof::AbstractString, thr::Real,
                                        args...; kwargs...)
    videomatch = match(FEATHER_VIDEO_REG, videof)
    videomatch === nothing && throw(ArgumentError("Could not parse $videof to find its datetime"))
    searchreg = sync_searchreg(videomatch)
    searchdir = splitdir(videof)[1]
    if isempty(searchdir)
        dirlisting = readdir()
    else
        dirlisting = readdir(searchdir)
    end
    syncf_ndx = findfirst(x -> occursin(searchreg, x), dirlisting)
    syncf_ndx === nothing && throw(error("Could not find sync file for $videof"))
    syncpath = joinpath(searchdir, dirlisting[syncf_ndx])
    feather_video_read_demin_audio(videof, syncpath, thr, args...; kwargs...)
end

exposure_map_grow!(grow_exposure_f::F, priv_data, img_buff) where F =
    grow_exposure_f(priv_data..., pop!(img_buff))

function exposure_map_drain!(grow_exposure_f::F, img_buff, priv_data) where F
    while !isempty(img_buff)
        priv_data = exposure_map_grow!(grow_exposure_f, priv_data, img_buff)
    end
    priv_data
end

function exposure_map_terminate!(end_exposure_f::F, priv_data, exposure_outs,
                                 exposed_ranges, img_buff, this_exposure_range) where F
    res = end_exposure_f(this_exposure_range, priv_data...)
    if exposure_outs === nothing
        exposure_outs = [res]
    else
        push!(exposure_outs, res)
    end
    push!(exposed_ranges, this_exposure_range)
    empty!(img_buff)
    exposure_outs
end

function exposure_map_loop_body!(new_exposure_f::F, grow_exposure_f::G, end_exposure_f::H, nexposed,
                                 in_steady_state, img_exposed, exposure_start, img_buff,
                                 priv_data, exposure_outs, exposed_ranges,
                                 img_cropped, shutter_delay, avg_norm, thr,
                                 frameno) where {F, G, H}

    frame_above_thr = frame_avg_intensity(img_cropped, avg_norm) >= thr
    nexposed = ifelse(frame_above_thr, nexposed + 1, 0)
    img_exposed = nexposed > shutter_delay
    if img_exposed
        steady_state = nexposed > 2 * shutter_delay
        if steady_state
            if !in_steady_state
                priv_data = new_exposure_f(img_cropped, priv_data...)
                exposure_start = frameno - shutter_delay
            end
            priv_data = exposure_map_grow!(grow_exposure_f, priv_data, img_buff)
            in_steady_state = true
        end
        isfull(img_buff) && error("circular buffer overflow")
        push!(img_buff, img_cropped)
    else # img not exposed
        if in_steady_state
            this_exposure_range = exposure_start : frameno - shutter_delay - 1
            exposure_outs = exposure_map_terminate!(end_exposure_f, priv_data,
                                    exposure_outs, exposed_ranges, img_buff,
                                    this_exposure_range)
            in_steady_state = false
        end
    end
    return nexposed, in_steady_state, img_exposed, exposure_start, priv_data, exposure_outs
end

"""
img_stack should not be normalized, nor should thr
"""
function exposure_map!(new_exposure_f::F, grow_exposure_f::G, end_exposure_f::H,
                       thr, img_stack, priv_data = ();
                       shutter_delay = 1,
                       roi_x::Union{Colon, <:UnitRange} = :,
                       roi_y::Union{Colon, <:UnitRange} = :) where {F, G, H}
    first_img, rest_imgs = peel(img_stack)
    roi_xr, roi_yr = check_slices(first_img, roi_x, roi_y)
    avg_norm = get_norm(roi_xr, roi_yr)
    img_cropped = view(first_img, roi_xr, roi_yr)
    img_buff = CircularBuffer{typeof(img_cropped)}(shutter_delay)
    nexposed = 0
    in_steady_state = false
    img_exposed = false
    exposure_start = -1
    curr_frameno = 1
    exposure_outs = nothing
    exposed_ranges = Vector{UnitRange{Int}}()
    nexposed, in_steady_state, img_exposed, exposure_start, priv_data, exposure_outs =
        exposure_map_loop_body!(new_exposure_f, grow_exposure_f, end_exposure_f,
                                nexposed, in_steady_state, img_exposed,
                                exposure_start, img_buff, priv_data,
                                exposure_outs, exposed_ranges, img_cropped,
                                shutter_delay, avg_norm, thr, curr_frameno)
    for img in rest_imgs
        curr_frameno += 1
        img_cropped = view(img, roi_xr, roi_yr)
        nexposed, in_steady_state, img_exposed, exposure_start, priv_data, exposure_outs =
            exposure_map_loop_body!(new_exposure_f, grow_exposure_f,
                                    end_exposure_f, nexposed, in_steady_state,
                                    img_exposed, exposure_start, img_buff,
                                    priv_data, exposure_outs, exposed_ranges,
                                    img_cropped, shutter_delay, avg_norm, thr,
                                    curr_frameno)
    end
    if img_exposed
        priv_data = exposure_map_drain!(grow_exposure_f, img_buff, priv_data)
        this_exposure_range = exposure_start : curr_frameno
        exposure_outs = exposure_map_terminate!(end_exposure_f, priv_data,
                                exposure_outs, exposed_ranges, img_buff,
                                this_exposure_range)
    end
    exposure_outs, exposed_ranges
end

function frames_min_max_accum_new_exp!(::Type{S}, img::AbstractArray{T};
                                       nt = nthreads()) where {S,T}
    nr, nc = size(img)
    rowrange = 1:nr
    if nt > 1
        blksize = cld(nc, nt)
        colranges = Vector{UnitRange{Int}}(undef, nt)
        @inbounds for tno in 1:nt
            lo = (tno - 1) * blksize + 1
            hi = min(tno * blksize, nc)
            colranges[tno] = lo:hi
        end
        tasks = Vector{Task}(undef, nt)
    else
        colranges = [1:nc]
        tasks = Vector{Task}()
    end
    minf, maxf, accf = frames_min_max_accum_alloc(S, T, size(img))
    frames_min_max_accum_init!(minf, maxf, accf)
    minf, maxf, accf, tasks, rowrange, colranges
end

function frames_min_max_accum_new_exp!(::DataType, ::AbstractArray, minf, maxf,
                                       accf, args...; kwargs...)
    frames_min_max_accum_init!(minf, maxf, accf)
    minf, maxf, accf, args...
end

function frames_min_max_accum_grow_exp!(minf, maxf, accf, tasks, rowrange,
                                        colranges, img)
    frame_min_max_accum!(minf, maxf, accf, tasks, img, rowrange, colranges)
    minf, maxf, accf, tasks, rowrange, colranges
end

function frames_min_max_accum_end_exp!(::Type{T}, exposed_range, minf, maxf,
                                       accf, args...; use_gamma = false) where T
    meanf = similar(accf, T)
    meanf .= accf ./ length(exposed_range)
    demeaned_minv = typemax(T)
    demeaned_maxv = typemin(T)
    @inbounds for i in eachindex(minf)
        meanv = meanf[i]
        demeaned_minv = min(demeaned_minv, minf[i] - meanv)
        demeaned_maxv = max(demeaned_maxv, maxf[i] - meanv)
    end
    outT = determine_container_depth(demeaned_maxv - demeaned_minv)
    maxscale = reinterpret(one(outT))
    f = make_scale_f(rawtype(outT), demeaned_minv, demeaned_maxv, maxscale, use_gamma)
    return outT, f, meanf
end

function feather_video_read_demean_write(videof, thr, framerate, outdir
                                         = pwd();
                                         scratch_dir = tempdir(),
                                         nt = nthreads(),
                                         use_gamma = false,
                                         roi_x::Union{Colon, <:UnitRange} = :,
                                         roi_y::Union{Colon, <:UnitRange} = :,
                                         shutter_delay = 1, force = false,
                                         kwargs...)
    imgs = reinterpret(UInt16,
                       convert_feather_video_frames(videof, scratch_dir = scratch_dir))
    img_stack = [view(imgs, :, :, i) for i in 1:size(imgs, 3)]
    new_exposure_f = (img, args...) -> frames_min_max_accum_new_exp!(UInt32,
                                                                     img,
                                                                     args...)
    end_exposure_f = (args...) -> frames_min_max_accum_end_exp!(Float32,
                                                                args...;
                                                                use_gamma = false)
    thr_scaled = thr * reinterpret(one(N6f10))
    outs, exposed_ranges = exposure_map!(new_exposure_f,
                                         frames_min_max_accum_grow_exp!,
                                         end_exposure_f, thr_scaled, img_stack;
                                         roi_x = roi_x, roi_y = roi_y,
                                         shutter_delay)
    for ((outT, f, meanf), exposed_range) in zip(outs, exposed_ranges)
        img_block = [view(imgs, roi_x, roi_y, j) for j in exposed_range]
        new_vid_path = joinpath(outdir,
                                demeaned_video_name(videof, exposed_range))
        write_demeaned_video(f, rawtype(outT), new_vid_path, img_block, meanf,
                             framerate; force, kwargs...)
    end
end

function feather_video_read_demean_grid_write(videof, thr, framerate, outdir
                                              = pwd();
                                              scratch_dir = tempdir(),
                                              nt = nthreads(),
                                              use_gamma = false,
                                              roi_x::Union{Colon, <:UnitRange} = :,
                                              roi_y::Union{Colon, <:UnitRange} = :,
                                              shutter_delay = 1, force = false,
                                              kwargs...)
    imgs = reinterpret(UInt16,
                       convert_feather_video_frames(videof, scratch_dir = scratch_dir))
    img_stack = [view(imgs, :, :, i) for i in 1:size(imgs, 3)]
    new_exposure_f = (img, args...) -> frames_min_max_accum_new_exp!(UInt32,
                                                                     img,
                                                                     args...)
    end_exposure_f = (args...) -> frames_min_max_accum_end_exp!(Float32,
                                                                args...;
                                                                use_gamma = false)
    thr_scaled = thr * reinterpret(one(N6f10))
    outs, exposed_ranges = exposure_map!(new_exposure_f,
                                         frames_min_max_accum_grow_exp!,
                                         end_exposure_f, thr_scaled, img_stack;
                                         roi_x = roi_x, roi_y = roi_y,
                                         shutter_delay)
    for ((outT, f, meanf), exposed_range) in zip(outs, exposed_ranges)
        img_block = [view(imgs, roi_x, roi_y, j) for j in exposed_range]
        new_vid_path = joinpath(outdir,
                                demeaned_video_name(videof, exposed_range))
        write_demeaned_grid_video(f, rawtype(outT), new_vid_path, img_block, meanf,
                             framerate; force, kwargs...)
    end
end

function demeaned_video_name(fname, exposed_range::AbstractUnitRange)
    bn, ext = splitext(basename(fname))
    "$(bn)_$(first(exposed_range))-$(last(exposed_range)).mp4"
end

function write_demeaned_video(f, ::Type{T}, out_filename, img_stack, meanf,
                              framerate; force = false,
                              container_private_settings =
                              (movflags = "+write_colr",),
                              encoder_settings = (color_range = 2,),
                              kwargs...) where T
    force_file_check(out_filename, force)
    first_img = first(img_stack)
    framebuff = similar(first_img, T)
    writer = open_video_out(out_filename, framebuff;
                             framerate, container_private_settings,
                             encoder_settings, scanline_major = true,
                             kwargs...)
    try
        for i in eachindex(img_stack)
            this_img = img_stack[i]
            for j in eachindex(this_img)
                framebuff[j] = f(this_img[j] - meanf[j])
            end
            append_encode_mux!(writer, framebuff, i - 1)
        end
    catch
        isfile(out_filename) && rm(out_filename, force = true)
        rethrow()
    finally
        close_video_out!(writer)
    end
    nothing
end

function write_demeaned_grid_video(f, ::Type{T}, out_filename, img_stack, meanf,
                                   framerate; force = false,
                                   kwargs...) where T<:UInt8
    force_file_check(out_filename, force)
    first_img = first(img_stack)
    framebuff = similar(first_img, T)
    gridbuff = MutableImage(RGB24, size(first_img))
    rgbbuff = get_image(gridbuff)
    writer = open_video_out(out_filename, framebuff; framerate,
                            scanline_major = true, kwargs...)
    try
        for i in eachindex(img_stack)
            this_img = img_stack[i]
            for j in eachindex(this_img)
                val = reinterpret(N0f8, f(this_img[j] - meanf[j]))
                rgbbuff[j] = convert(RGB24, Gray(val))
            end
            grid_lines(gridbuff)
            framebuff .= reinterpret(T, convert.(Gray{N0f8}, get_image(gridbuff)))
            append_encode_mux!(writer, framebuff, i - 1)
        end
    catch
        isfile(out_filename) && rm(out_filename, force = true)
        rethrow()
    finally
        close_video_out!(writer)
    end
    nothing
end

function streaming_intensities(fname, roi_x = :, roi_y = :; nt = nthreads())
    outs = frame_iter_preamble(fname)
    outs === nothing && return Float64[]
    r, img_raw = outs
    roi_xr, roi_yr = check_slices(img_raw, roi_x, roi_y)
    norm = get_norm(roi_xr, roi_yr)

    nframes = get_number_frames(fname)
    if nframes === nothing
        error("Could not find the number of frames from video container $fname")
    end

    intensities = Vector{Float64}(undef, nframes)
    fno = 1
    @inbounds intensities[fno] = frame_avg_intensity(convert_featherscope_rgb,
                                                     UInt64, img_raw, norm;
                                                     roi_xr, roi_yr, nt)
    while !eof(r) && fno < nframes
        read!(r, img_raw)
        fno += 1
        @inbounds intensities[fno] = frame_avg_intensity(convert_featherscope_rgb,
                                                         UInt64, img_raw, norm;
                                                         roi_xr, roi_yr, nt)
    end
    fno < nframes && resize!(intensities, fno)
    return intensities
end

function update_exposure_first_frame(ignore_exposure, intensity, thr)
    above_thr = intensity >= thr
    # if intensity falls below thr, ignore_exposure should stay false
    ignore_exposure &= above_thr
    is_exposed = !ignore_exposure & above_thr
    ignore_exposure, is_exposed
end

"""
    find_first_exposed_frame(fname, thr, roi_x = :, roi_y = : ;
                             nt = nthreads(), shutter_delay = 1,
                             skip_exposure_at_start = true) -> Union{Int, Nothing}

Find the average intensity in the roi of successive frames of `fname`, and
return the frame number where the intensity is greater than or equal to `thr`.
If `fname` is empty, or if all frames have an average intensity in the roi less
than `thr`, then return `nothing`.
"""
function find_first_exposed_frame(fname, thr, roi_x = :, roi_y = : ;
                                  nt = nthreads(), shutter_delay = 1,
                                  skip_exposure_at_start = true)
    outs = frame_iter_preamble(fname)
    outs === nothing && return nothing
    r, img_raw = outs
    roi_xr, roi_yr = check_slices(img_raw, roi_x, roi_y)
    norm = get_norm(roi_xr, roi_yr)

    fno = 1
    intensity = frame_avg_intensity(convert_featherscope_rgb,
                                    UInt64, img_raw, norm; roi_xr, roi_yr, nt)
    ignore_exposure, is_exposed = update_exposure_first_frame(skip_exposure_at_start,
                                                              intensity, thr)
    nexposed = ifelse(is_exposed, 1, 0)
    while !eof(r) & (nexposed <= shutter_delay)
        fno += 1
        read!(r, img_raw)
        intensity = frame_avg_intensity(convert_featherscope_rgb, UInt64,
                                        img_raw, norm; roi_xr, roi_yr, nt)
        ignore_exposure, is_exposed = update_exposure_first_frame(ignore_exposure,
                                                                  intensity, thr)
        nexposed = ifelse(is_exposed, nexposed + 1, 0)
    end
    return eof(r) ? nothing : fno
end

function find_exposed_frame_ranges(fname, thr, roi_x = :, roi_y = : ;
                                   nt = nthreads(), shutter_delay = 1,
                                   skip_exposure_at_start = true)
    out = Vector{UnitRange{Int}}()
    frame_info = frame_iter_preamble(fname)
    frame_info === nothing && return out
    r, img_raw = frame_info
    roi_xr, roi_yr = check_slices(img_raw, roi_x, roi_y)
    norm = get_norm(roi_xr, roi_yr)

    fno = 1
    intensity = frame_avg_intensity(convert_featherscope_rgb, UInt64, img_raw,
                                    norm; roi_xr, roi_yr, nt)
    ignore_exposure, is_exposed = update_exposure_first_frame(skip_exposure_at_start,
                                                              intensity, thr)
    nexposed = ifelse(is_exposed, 1, 0)
    while !eof(r)
        fno += 1
        read!(r, img_raw)
        intensity = frame_avg_intensity(convert_featherscope_rgb, UInt64,
                                        img_raw, norm; roi_xr, roi_yr, nt)
        ignore_exposure, is_exposed = update_exposure_first_frame(ignore_exposure,
                                                                  intensity, thr)
        if !is_exposed & (nexposed > shutter_delay) # Leaving exposure
            last_exposed = fno - 1
            push!(out, last_exposed - nexposed + shutter_delay + 1:
                  last_exposed - shutter_delay)
        end
        nexposed = ifelse(is_exposed, nexposed + 1, 0)
    end
    if nexposed > shutter_delay # Unfinished exposure
        # Do not account for the delay in the shutter closing at the end of file
        push!(out, fno - nexposed + shutter_delay + 1: fno)
    end
    out
end

function avi_to_scaled_gray_video(out_filename, in_filename, framerate;
                                  scratch_dir = "", nt = nthreads(),
                                  use_gamma = false, force = false,
                                  container_private_settings =
                                  (movflags = "+write_colr",),
                                  encoder_settings = (color_range = 2,),
                                  kwargs...)
    encoder_settings = (color_range = 2,),
    force_file_check(out_filename, force)
    imgs = convert_feather_video_frames(in_filename; scratch_dir)
    nx, ny, nf = size(imgs)
    minv, maxv = extrema(imgs)
    l = make_pixel_lut(minv, maxv, one(N6f10), use_gamma)
    framebuff = Matrix{N6f10}(undef, nx, ny)
    writer = open_video_out(out_filename, framebuff; framerate,
                             scanline_major = true, container_private_settings,
                             encoder_settings, kwargs...)
    for i in 1:nf
        apply_lut!(l, framebuff, view(imgs, :, :, i); nt)
        append_encode_mux!(writer, framebuff, i - 1)
    end
    close_video_out!(writer)
    nothing
end

function combine_featherscope_chunks(files, outfile = tempname(); roi_x = :,
                                     roi_y = :)
    for file in files
        isfile(file) ||
            throw(ArgumentError("File $file does not exist"))
    end
    sz = nothing
    img_raw = nothing
    gray_img = nothing
    frame_ranges = Vector{UnitRange{Int}}()
    open(outfile, "w") do io
        for file in files
            r = openvideo(file)
            eof(r) && error("No video in $file")
            if img_raw === nothing
                img = read(r)::PermutedDimsArray{RGB{Normed{UInt8,8}},2,(2, 1),(2, 1),Array{RGB{Normed{UInt8,8}},2}}
                img_raw = parent(img)
            else
                img_raw = read!(r, img_raw)
            end
            roi = check_slices(img_raw, roi_x, roi_y)
            this_sz = length.(roi)
            if sz === nothing
                sz = this_sz
            else
                sz == this_sz || error("sizes are not the same")
            end
            if gray_img === nothing
                gray_img = similar(img_raw, UInt16, this_sz)
            end
            gray_img .= convert_featherscope_rgb.(view(img_raw, roi...))
            first_frame = isempty(frame_ranges) ? 1 : last(last(frame_ranges)) + 1
            fno = first_frame
            write(io, gray_img)
            while !eof(r)
                read!(r, img_raw)
                gray_img .= convert_featherscope_rgb.(view(img_raw, roi...))
                write(io, gray_img)
                fno += 1
            end
            push!(frame_ranges, first_frame : fno)
        end
    end
    return outfile, sz, frame_ranges
end

function time_range(vidfile)
    start_time = parse_bonsai_timestr((vidfile))
    start_time === nothing && return
    dur_secs = VideoIO.get_duration(vidfile)
    stop_time = start_time + Millisecond(round(Int, 1000 * dur_secs))
    start_time, stop_time
end

function parse_bonsai_timestr(s)
    df = dateformat"Y-m-dTH_M_S"
    dreg = r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})"
    m = match(dreg, s)
    match === nothing && return
    DateTime(m[1], df)
end

end # module
