function find_sync_edges(sync_pulses::AbstractVector{<:Number},
                         sync_high::Number = FEATHER_SYNC_HIGH,)
    find_all_edge_triggers(sync_pulses, sync_high / 2)
end

find_sync_edges(sync_data::AbstractMatrix, args...) =
    find_sync_edges(view(sync_data, 2, :), args...)

function find_shutter_edges(shutter_signal::AbstractVector{<:Number},
                               shutter_high = FEATHER_SHUTTER_HIGH)
    find_all_edge_triggers(shutter_signal, shutter_high / 2)
end

find_shutter_edges(sync_data::AbstractMatrix, args...) =
    find_shutter_edges(view(sync_data, 3, :), args...)

"""
    measure_framerate(sync_edge_ndxs::Vector{Int}, fs_sync::Float64[,
                      framerate_precision:Rational]) -> Float64, Rational{Int}

Determine the framerate of a video from the indices of its sync pulses,
`sync_edge_ndxs`, and the sampling rate, `fs_sync`. If `framerate_precision` is
not specified, then the framerate will be a `Float64`, otherwise it will be
rounded to the nearest multiple of `framerate_precision`, and returned as a
`Rational{Int}`.
"""
measure_framerate(sync_edge_ndxs::AbstractVector{<:Number}, fs_sync::Number) =
    fs_sync / mean(diff(sync_edge_ndxs))

function measure_framerate(sync_edge_ndxs, fs_sync, framerate_precision::Rational)
    framerate_f = measure_framerate(sync_edge_ndxs, fs_sync)
    framerate_num = round(Int, denominator(framerate_precision) * framerate_f)
    framerate = framerate_num // denominator(framerate_precision)
    return framerate
end

function shutter_sync_edges(syncdata, shutter_thr = 0.1, sync_thr = 0.1)
    shutter_edges = find_shutter_edges(syncdata, shutter_thr)
    sync_edges = find_sync_edges(syncdata, sync_thr)
    return shutter_edges, sync_edges
end

"""
    video_sync_alignment(shutter_edges::Vector{Int},
                         sync_edges::Vector{Int},
                         sync_exposed_frameno::Int,
                         crossingno::Int,
                         framerate::AbstractFloat;
                         fs_sync::Float64 = 48000.0,
                         shutter_skip_frames::Int = 1) -> Union{Nothing, Float64}

    video_sync_alignment(syncdata::Matrix{Float64}, ...;
                         shutter_thr = 0.1, sync_thr = 0.1, ...)


Find the seconds of delay between the start of the sync file and the start of
the video file. Negative values indicate that the sync file starts after the
video file.
"""
function video_sync_alignment(shutter_edges::AbstractVector{<:Integer},
                              sync_edges::AbstractVector{<:Integer},
                              sync_exposed_frameno::Integer,
                              crossingno::Integer,
                              framerate::Number;
                              fs_sync::Number = 48000.0,
                              shutter_skip_frames::Integer = 1,
                              sync_jitter_s::Number = 0.002)

    if length(shutter_edges) < crossingno
        throw(ArgumentError("shutter_edges must be longer than crossingno"))
    end
    # Find the time from the beginning of the original video file of the first
    # exposed frame that can be sync'd
    video_exposure_time = ndx_to_t(sync_exposed_frameno, framerate)

    shutter_ndx = shutter_edges[crossingno]

    # Find the time from the beginning of the sync file of the corresponding
    # shutter opening, and use it to find when the sync file starts, relative to
    # the start of the video file.
    if isempty(sync_edges)
        # cannot work with sync, due to experimental error it is empty
        shutter_open_time = ndx_to_t(shutter_ndx, fs_sync)
        shutter_exposure_time = shutter_open_time + shutter_skip_frames / framerate
        rel_sync_start_time = shutter_exposure_time - video_exposure_time
    else
        jittered_shutter_ndx = shutter_ndx - round(Int, sync_jitter_s * fs_sync)
        sync_shutter_no = searchsortedfirst(sync_edges, jittered_shutter_ndx)
        sync_exposure_no = sync_shutter_no + shutter_skip_frames
        sync_exposure_no > length(sync_edges) && return
        sync_exposure_time = ndx_to_t(sync_edges[sync_exposure_no], fs_sync)
        rel_sync_start_time = sync_exposure_time - video_exposure_time
    end

    return rel_sync_start_time
end

function video_sync_alignment(syncdata::AbstractMatrix{<:AbstractFloat},
                              args...; shutter_thr = 0.1, sync_thr = 0.1,
                              fs_sync = 48000.0, kwargs...)
    shutter_edges, sync_edges = shutter_sync_edges(syncdata, shutter_thr, sync_thr)
    video_sync_alignment(shutter_edges, sync_edges, args...; kwargs...)
end

function sync_sanity_check(sync_edges, syncl)
    sync_pers = diff(sync_edges)
    med_per = median(sync_pers)
    extreme_per_rat = maximum(abs.(extrema(sync_pers) ./ med_per .- 1))
    leading_gap = max(first(sync_edges) - 1 - med_per, 0) / med_per
    trailing_gap = max(syncl - last(sync_edges) - med_per, 0) / med_per
    return extreme_per_rat, leading_gap, trailing_gap
end

function normalized_tsdiffs(ts)
    tsdiffs = diff(ts)
    meddif = median(tsdiffs)
    outlier_rats = tsdiffs ./ meddif .- 1
    return outlier_rats
end

"""
    outlier_is_compensated(ts, pos, maxnframe = 5; zero_tol = 0.1, sum_tol = 0.2) -> Bool

Determine if a large delay between timestamps is compensated by a series of
short delays between timestamps, or if instead a frame has been dropped. Given
an array of normalized differences between successive timestamps, `ts_diffs`,
and the position of the large positive outlier, `pos`, advance up to `maxnframe`
from this initial position, and determine if the sum of these normalized
differences comes within `sum_tol` of zero. Will also stop the if a normalized
difference comes within `zero_tol` of zero.
"""
function outlier_is_compensated(ts_diffs, pos, maxnframe = 5; zero_tol = 0.1, sum_tol = 0.2)
    ts_diff_len = length(ts_diffs)
    @boundscheck 1 <= pos <= ts_diff_len || return false
    @inbounds running_sum = ts_diffs[pos]
    for i in pos+1:min(ts_diff_len, pos + maxnframe)
        # Stop running sum if timestamps have returned to baseline
        val = ts_diffs[i]
        if val >= -zero_tol || running_sum <= sum_tol
            break
        end
        running_sum += val
    end
    return running_sum <= sum_tol
end

"Check quality of featherscope files, synchronize them, and calculate framerate"
function triplet_sync_info(vfname, sfname, tfname;
                           fs_sync = 48000.0, shutter_thr = 0.1, sync_thr = 0.1,
                           exposure_thr = 0.04, x = :, y = :,
                           framerate_precision = 1//100,
                           def_framerate = 741//25, sync_rat_tol = 0.05,
                           timestamp_rat_tol = 0.8,
                           skip_exposure_at_start = false,
                           kwargs...)
    syncdata = open_audio_sync(sfname)
    shutter_edges, sync_edges = shutter_sync_edges(syncdata, shutter_thr, sync_thr)
    triplet_ok = true
    if isempty(sync_edges)
        @warn "Could not find sync edges for file $sfname"
        framerate = def_framerate
        framerate_out = nothing
    else
        syncl = size(syncdata, 2)
        extreme_per_rat, leading_gap, trailing_gap =
            sync_sanity_check(sync_edges, syncl)
        if any(x -> x > sync_rat_tol, extreme_per_rat)
            @warn "Sync edges for $sfname fail quality control"
            empty!(sync_edges)
            framerate = def_framerate
            framerate_out = nothing
        else
            framerate = measure_framerate(sync_edges, fs_sync, framerate_precision)
            framerate_out = framerate
        end
    end

    ts = open_utinstants(tfname)
    if isempty(ts)
        @warn "$tfname is empty"
        triplet_ok = false
    else
        normed_tsdiffs = normalized_tsdiffs(ts)
        for extr_pos in findall(x -> x >= timestamp_rat_tol, normed_tsdiffs)
            if !outlier_is_compensated(normed_tsdiffs, extr_pos)
                @warn "Timestamps for $tfname fail quality control"
                triplet_ok = false
                break
            end
        end
        nts = length(ts)
        nframe = get_number_frames(vfname)
        if nts != nframe
            triplet_ok = false
            @warn "Number of video frames does match number of timestamps for $vfname"
        end
    end
    sync_exposed_frameno, crossingno, triplet_ok = try
        sync_exposed_frameno, crossingno = find_first_exposure_edge(vfname, exposure_thr, x, y)
        sync_exposed_frameno, crossingno, triplet_ok
    catch err
        if err isa ErrorException && occursin("Could not open", err.msg)
            @warn "Could not open video $vfname"
            @warn "Error was:"
            showerror(stderr, err, catch_backtrace())
            nothing, false
        else
            rethrow()
        end
    end
    if sync_exposed_frameno === nothing
        @warn "Could not find shutter opening for file $vfname"
        rel_sync_start_time = nothing
        triplet_ok = false
    else
        rel_sync_start_time = video_sync_alignment(shutter_edges, sync_edges,
                                                   sync_exposed_frameno,
                                                   crossingno, framerate;
                                                   fs_sync = fs_sync,
                                                   kwargs...)
        if rel_sync_start_time === nothing
            @warn "video_sync_alignment returned nothing for $vfname"
            triplet_ok = false
        end
    end
    return rel_sync_start_time, framerate_out, triplet_ok
end

"Call `triplet_sync_info` on a group of files"
function sync_triplets(trips; kwargs...)
    nt = length(trips)
    vid_offsets = Vector{Union{Nothing, Float64}}(undef, nt)
    framerates = Vector{Union{Nothing, Rational{Int}}}(undef, nt)
    triplets_ok = Vector{Union{Nothing, Bool}}(undef, nt)
    for (i, (vf, sf, tf)) in enumerate(trips)
        @info "Synchronizing $vf"
        out = triplet_sync_info(vf, sf, tf; kwargs...)
        if out === nothing
            @inbounds vid_offsets[i] = framerates[i] = triplets_ok[i] = nothing
        else
            @inbounds vid_offsets[i], framerates[i], triplets_ok[i] = out
        end
    end
    return vid_offsets, framerates, triplets_ok
end
