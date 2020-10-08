module FeatherScopePrePro

using CaProcessing: clip_segments_thr, clip_imgs, demin, map_to_8bit, PixelLUT,
    pixel_lut, rescale_compress
using GLUtilities: ndx_to_t, t_to_ndx, clip_ndx
using ColorTypes: Gray, RGB
using DataStructures: OrderedDict

using FeatherscopeExtraction: convert_feather_video_frames, check_slices,
    convert_featherscope_rgb, feather_video_min_frame_planning, open_audio_sync,
    FEATHER_VIDEO_REG, FEATHER_SYNC_REG, sync_searchreg, video_sync_alignment

using FileIO: save
using FixedPointNumbers: Normed # need reinterpret method
using ImageCore: colorview
using JSON: json
using WAV: wavwrite

using VideoIO: prepareencoder, VideoEncoder, openvideo, appendencode!,
    finishencode!, mux
import VideoIO

import FFMPEG

export avis_to_tiff_demin,
    avi_to_tiff_demin,
    avi_to_tiff_raw,
    feather_video_read_demin_audio,
    feather_video_encode_demind_segments

const AVI_REGEX = r"(?<file_prefix>.*)\.avi$"i
const JSON_DICT_TYPE = OrderedDict{String, Any}

function avis_to_tiff_demin(savedir, fnames, x, y, thr;
                      scratch = tempdir(), nt = Threads.nthreads())
    for fname in fnames
        avi_to_tiff_demin(savedir, fname, x, y, thr, scratch = scratch)
    end
end

function avi_to_tiff_demin(savedir::AbstractString, fname::AbstractString,
                           x::AbstractRange, y::AbstractRange, thr::Real;
                           scratch = tempdir(), kwargs...)
    imgs = convert_feather_video_frames(fname, parentdir = scratch)
    avi_to_tiff_demin(savedir, imgs, fname, x, y, thr; kwargs...)
end

function avi_to_tiff_demin(savedir::AbstractString, imgs::AbstractArray,
                           fname::AbstractString, x::AbstractRange,
                           y::AbstractRange, thr::Real;
                           nt = Threads.nthreads(),
                           name_f = default_name_conversion)
    # convert video and find exposed segments
    imgs_roi = clip_imgs(imgs, x = x, y = y)
    segs, open_pers = clip_segments_thr(imgs_roi, thr, nt = nt)

    nseg = length(segs)
    nframes = [open_pers[2, i] - open_pers[1, i] + 1 for i in 1:nseg]
    keep_idxs = findall(x -> x > 1, nframes)
    nkeep = length(keep_idxs)

    # build metadata json
    pref = extract_avi_prefix(fname)
    json_fname = joinpath(savedir, pref * ".json")

    json_dict = JSON_DICT_TYPE()
    json_dict["working_dir"] = pwd()
    json_dict["input_avi"] = fname
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

function  avi_to_tiff_raw(savedir, fname; scratch = tempdir())
    pref = extract_avi_prefix(fname)
    imgs = convert_feather_video_frames(fname, parentdir = scratch)
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

struct FrameEncoderState
    lut::PixelLUT{UInt8}
    io::IOStream
    s_fpath::String
    encoder::VideoEncoder
end

function start_encode(writedir, graybuf, framerate, props)
    s_fpath = joinpath(writedir, "temp.stream")
    io = open(s_fpath, "w")
    writebuf = PermutedDimsArray(graybuf, (2,1))
    encoder = prepareencoder(writebuf, framerate = framerate,
                             AVCodecContextProperties = props)
    return io, s_fpath, encoder
end

function finish_encode(encoder_state, writedir, new_fname, exposed_range, framerate)
    finishencode!(encoder_state.encoder, encoder_state.io)
    close(encoder_state.io)
    new_fpath = joinpath(writedir, new_fname)
    mux(encoder_state.s_fpath, new_fpath, framerate)
end

function append_frame!(encoder_state, graybuf, fno)
    writebuf = PermutedDimsArray(graybuf, (2,1))
    appendencode!(encoder_state.encoder, encoder_state.io, writebuf, fno)
end

function append_demind_video_frame!(encoder_state, exposure_no, graybuf, img_raw,
                                    exposed_range, min_frame, sub_maxv, fno,
                                    roi_xr, roi_yr, writedir, new_fname,
                                    framerate, props)
    if encoder_state === nothing
        if fno in exposed_range
            io, s_fpath, encoder = start_encode(writedir, graybuf, framerate, props)
            lut = pixel_lut(x -> rescale_compress(UInt8, x, 1/reinterpret(sub_maxv)),
                            reinterpret(sub_maxv))
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
            finish_encode(encoder_state, writedir, new_fname, exposed_range, framerate)
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

function feather_video_encode_demind_segments(input_fname, roi_x, roi_y, min_frames,
                                              subtracted_maxvals, exposed_ranges,
                                              framerate, props, writedir = pwd();)
    nexposure = length(exposed_ranges)
    nexposure > 0 || return String[]
    isdir(writedir) || throw(ArgumentError("Cannot access write directory $writedir"))
    base_name = basename(input_fname)
    video_names = def_fname_f.(base_name, exposed_ranges)
    inputio = VideoIO.open(input_fname)
    outs = try
        inputvid = openvideo(inputio)
        try
            eof(inputvid) && return String[]
            img_raw_p = VideoIO.read(inputvid)::PermutedDimsArray{RGB{Normed{UInt8,8}},2,(2, 1),(2, 1),Array{RGB{Normed{UInt8,8}},2}}
            img_raw = parent(img_raw_p)
            roi_xr, roi_yr = check_slices(img_raw, roi_x, roi_y)
            nxr = length(roi_xr)
            nyr = length(roi_yr)
            graybuf = Matrix{UInt8}(undef, nxr, nyr)
            encoder_state = nothing
            fno = 1
            exposure_no = 1
            encoder_state, exposure_no = append_demind_video_frame!(
                encoder_state, exposure_no, graybuf, img_raw,
                exposed_ranges[exposure_no], min_frames[exposure_no],
                subtracted_maxvals[exposure_no], fno, roi_xr, roi_yr, writedir,
                video_names[exposure_no], framerate, props
            )
            while !eof(inputvid) && exposure_no <= nexposure
                VideoIO.read!(inputvid, img_raw)
                fno += 1
                encoder_state, exposure_no = append_demind_video_frame!(
                    encoder_state, exposure_no, graybuf, img_raw,
                    exposed_ranges[exposure_no], min_frames[exposure_no],
                    subtracted_maxvals[exposure_no], fno, roi_xr, roi_yr,
                    writedir, video_names[exposure_no], framerate,
                    props
                )
            end
            if encoder_state !== nothing
                finish_encode(encoder_state, writedir, video_names[exposure_no],
                              exposed_ranges[nexposure], framerate)
            end
        finally
            close(inputvid)
        end
    finally
        close(inputio)
    end
    return joinpath.(writedir, video_names)
end

function feather_video_read_demin(input_fname, thr, roi_x, roi_y, framerate,
                                  props, writedir = pwd())
    min_frames, subtracted_maxvals, exposed_ranges, nf =
        feather_video_min_frame_planning(input_fname, thr, roi_x, roi_y)
    feather_video_encode_demind_segments(input_fname, roi_x, roi_y, min_frames,
                                         subtracted_maxvals, exposed_ranges,
                                         framerate, props, writedir)
end

function feather_sync_add_audio(syncf, new_fnames, exposed_ranges, sync_frameno,
                                framerate, writedir, shutter_offset, fs_sync,
                                force_video, nexposed; kwargs...)
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
        # Make audio file
        wavwrite(audio_segment, temp_audio_f, Fs = fs_sync)
        try
            mv(new_fnames[exposure_no], temp_video_f, force = force_video)
            try
                # join video and audio
                FFMPEG.exe(`-y -i $(temp_video_f) -i $(temp_audio_f) -c:v copy
                            -c:a aac -map 0:v:0 -map 1:a:0
                            $(new_fnames[exposure_no])`)
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
                                        roi_y, framerate, props,
                                        writedir = pwd(); shutter_offset = 1,
                                        fs_sync = 48000, force_video = false,
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

    new_fnames = feather_video_encode_demind_segments(videof, roi_x, roi_y,
                                                      min_frames,
                                                      subtracted_maxvals,
                                                      exposed_ranges, framerate,
                                                      props, writedir)
    sync_exposed_frameno = first_exposure_nosync ? exposed_ranges[2][1] :
                                                   exposed_ranges[1][1]
    try
        feather_sync_add_audio(syncf, new_fnames, exposed_ranges,
                               sync_exposed_frameno, framerate, writedir,
                               shutter_offset, fs_sync, force_video, nexposed;
                               kwargs...)
    catch
        for f in new_fnames
            isfile(f) && rm(f)
        end
        rethrow()
    end
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

end # module
