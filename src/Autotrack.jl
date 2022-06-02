module Autotrack

using Dates, LinearAlgebra, Statistics
using VideoIO, ColorVectorSpace, Dierckx, ImageTransformations, ImageDraw, StaticArrays, PaddedViews, OffsetArrays, ImageFiltering, ColorTypes, CSV
SV = SVector{2, Float64}

export track, track2csv

# include("video.jl")
# sz = (108, 144)


# _parse_time(::Nothing, T) = 0.0
# _parse_time(x, T) = Millisecond(T(parse(Int, x)))/Millisecond(Second(1))
# _parse_time(xT) = _parse_time(xT...)
#
# function _get_seconds(txt)
#   m = match(r"(?:(\d+):)*(\d+):(\d+)(?:,(\d+))*", txt)
#   sum(_parse_time, zip(m.captures, (Hour, Minute, Second, Millisecond)))
# end

function seekread!(img, vid, t)
  seek(vid, t)
  read!(vid, img)
end

# function get_times(start::Time, stop::Time)
#   t1 = _get_seconds(start)
#   t2 = _get_seconds(stop)
#   range(t1, t2, length = nframes)
# end

function get_next(guess, img, bkgd, wr)
  centered_img = OffsetArrays.centered(img, Tuple(guess))
  centered_bkgd = OffsetArrays.centered(bkgd, Tuple(guess))
  w = CartesianIndex(wr, wr)
  window = -w:w
  x = centered_bkgd[window] .- centered_img[window]
  imfilter!(x, x, Kernel.DoG(0.85))
  _, i = findmax(x)
  guess + window[i]
end

function get_imgs(vid, ts, spatial_step, nframes, wr)
  img = read(vid)
  # t₀ = gettime(vid)
  h, w = size(img)
  width_ind = 1:spatial_step:w
  height_ind = 1:spatial_step:h
  sz = (length(height_ind), length(width_ind))

  unpadded_imgs = [similar(img, sz) for _ in 1:nframes]
  for (i, t) in enumerate(ts)
    seekread!(img, vid, t)
    # seekread!(img, vid, t + t₀)
    unpadded_imgs[i] .= img[height_ind, width_ind]
  end

  height, width = sz
  padded_axes = (1-wr:height+wr, 1-wr:width+wr)
  imgs = PaddedView.(zero(eltype(img)), unpadded_imgs, Ref(padded_axes))
  bkgd = PaddedView(zero(eltype(img)), mean(unpadded_imgs), padded_axes)

  return sz, imgs, bkgd
end

function get_spline(imgs, bkgd, ts, guess, smoothing_factor, wr)
  coords = accumulate((guess, img) -> get_next(guess, img, bkgd, wr), imgs, init = guess)
  ParametricSpline(ts, hcat(SV.(Tuple.(coords))...); s = smoothing_factor, k = 2)
end

get_guess(::Missing, sz, _) = CartesianIndex(sz .÷ 2) 
get_guess(starting_point, _, spatial_step) = CartesianIndex(starting_point .÷ spatial_step)

function track(file::AbstractString, start_time::Real, stop_time::Real; 
    csv_file::Union{Nothing, AbstractString} = "tracked", debug_file::Union{Nothing, AbstractString} = "tracked", starting_point::Union{Missing, CartesianIndex{2}} = missing, temporal_step = 2.0, spatial_step = 10, smoothing_factor = 200, window_radius = 4)

  vid = VideoIO.openvideo(file, target_format=VideoIO.AV_PIX_FMT_GRAY8)
  ts = range(start_time, stop_time, step = temporal_step)
  nframes = length(ts)
  sz, imgs, bkgd = get_imgs(vid, ts, spatial_step, nframes, window_radius)
  guess = get_guess(starting_point, sz, spatial_step)
  spl = get_spline(imgs, bkgd, ts, guess, smoothing_factor, window_radius)

  ar = VideoIO.aspect_ratio(vid)

  aspect_ratio = spatial_step*[1, ar]

  savevid(debug_file, ts, spl, imgs)

  savecsv(csv_file, ts, spl, aspect_ratio)

  return ts[1], ts[end], spl, aspect_ratio
end

function _fun(t, spl, ar)
  x, y = ar .* spl(t)
  return (; t, x, y)
end

savecsv(::Nothing, args...) = nothing

savecsv(csv_file, ts, spl, aspect_ratio) = CSV.write(csv_file * ".csv", [_fun(t, spl, aspect_ratio) for t in ts])

savevid(::Nothing, args...) = nothing

function savevid(result_file, ts, spl, imgs)
  # ar = SV(1, aspect_ratio)
  # txy = (NamedTuple{(:t, :x, :y)}((t, ar .* spl(t)...)) for t in ts[1] : 1/10 : ts[end])
  # # txy = (NamedTuple{(:t, :x, :y)}((t, spl(t)...)) for t in ts[1] : 1/10 : ts[end])
  # CSV.write("$result_file.csv", txy)

  encoder_options = (color_range=2, crf=23, preset="medium")
  time = 2
  framerate = round(Int, length(ts)/time)
  open_video_out("$result_file.mp4", parent(imgs[1]), framerate=framerate, encoder_options=encoder_options) do writer
    for (i, t) in enumerate(ts)
      path = Path([CartesianIndex(Tuple(round.(Int, spl(i)))) for i in ts[1] : 1/10 : t])
      img = parent(imgs[i])
      draw!(img, path)
      write(writer, img)
    end
  end

end

# function process_track(video_file, start, stop, result_file)
#   data = get_data(video_file, start, stop)
#   save(result_file, data...)
# end
#
# function _process_tracks(rows, f, video_dir, result_dir)
#   @showprogress for (row, (file, start, stop)) in enumerate(f)
#     process_track(joinpath(video_dir, file), start, stop, joinpath(result_dir, string(row)))
#   end
# end
#
# function process_tracks(csvfile; video_dir = dirname(csvfile), result_dir = dirname(csvfile))
#   f = CSV.File(csvfile)
#   rows = 1:length(f)
#   _process_tracks(rows, f, video_dir, result_dir)
# end
#
# function process_tracks(csvfile, row::Int; video_dir = dirname(csvfile), result_dir = dirname(csvfile))
#   f = CSV.File(csvfile)
#   file, start, stop = f[row]
#   process_track(joinpath(video_dir, file), start, stop, joinpath(result_dir, string(row)))
# end
#
# function process_tracks(csvfile, rows::AbstractVector{Int}; video_dir = dirname(csvfile), result_dir = dirname(csvfile))
#   f = CSV.File(csvfile)
#   _process_tracks(rows, f, video_dir, result_dir)
# end

end
