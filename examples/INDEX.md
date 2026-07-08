# TEX Examples Index

> **Generated** by `tools/gen_examples_index.py` from the `// Name — desc`
> header of each `examples/*.tex`. 114 examples. Do not edit by hand.

| Example | Description |
|---------|-------------|
| [Alpha Over](examples/alpha_over.tex) | composite foreground over background using a mask |
| [Array Reduce](examples/array_reduce.tex) | demonstrate array aggregation functions |
| [Auto Levels](examples/auto_levels.tex) | normalize image channels to the full [0,1] range |
| [Barrel Distortion](examples/barrel_distortion.tex) | polynomial radial lens distortion |
| [Bilateral Approx](examples/bilateral_approx.tex) | edge-preserving blur (approximate bilateral filter) |
| [Billow Texture](examples/billow_texture.tex) | Puffy, cloud-like procedural noise |
| [Binding Access](examples/binding_access.tex) | brackets fetch (nearest), parens sample (bilinear) |
| [Blur](examples/blur.tex) | simple box blur using fetch() |
| [Box Blur](examples/box_blur.tex) | Simple NxN averaging kernel |
| [Break Search](examples/break_search.tex) | scan for the first bright pixel using for + break |
| [Brightness / Contrast](examples/brightness_contrast.tex) | filmic exposure and contrast adjustment |
| [Caustics](examples/caustics.tex) | scatter-write underwater light simulation |
| [Channel Shuffle](examples/channel_shuffle.tex) | remap channels between two inputs |
| [Channel Swap](examples/channel_swap.tex) | swap red and blue channels |
| [Chroma Keyer](examples/chroma_keyer.tex) | color-difference keyer with spill suppression |
| [Radial chromatic aberration](examples/chromatic_aberration.tex) | lens-style RGB fringing |
| [Color Functions](examples/color_functions.tex) | reusable function library for color grading |
| [Color Grade](examples/color_grade.tex) | lift / gamma / gain with saturation |
| [Color Mix](examples/color_mix.tex) | blend two images using a mask |
| [Composite](examples/composite.tex) | alpha-over compositing with mask |
| [Conditional](examples/conditional.tex) | warm shadows, cool highlights |
| [Const Values](examples/const_values.tex) | demonstrate const declarations for color grading |
| [Convolve](examples/convolve.tex) | blur an image using a second image as the kernel |
| [Corner Pin](examples/corner_pin.tex) | bilinear warp using four corner positions |
| [Curl Distortion](examples/curl_distortion.tex) | Divergence-free UV warping with curl noise |
| [Custom Blend](examples/custom_blend.tex) | overlay blend mode using user-defined functions |
| [Denoise](examples/denoise.tex) | Non-Local Means (NLM) denoiser |
| [Difference Key](examples/difference_key.tex) | extract a matte from the difference between two images |
| [Directional Blur](examples/directional_blur.tex) | motion blur along a specific angle |
| [Distortion Map](examples/distortion_map.tex) | offset-based displacement |
| [Edge Detect](examples/edge_detect.tex) | Sobel gradient magnitude |
| [Emboss](examples/emboss.tex) | directional emboss via offset sampling |
| [Erode / Dilate](examples/erode_dilate.tex) | shrink or grow mask channels |
| [Fast Blur](examples/fast_blur.tex) | mipmap-accelerated constant-time blur |
| [Fast Defocus](examples/fast_defocus.tex) | mipmap-accelerated disc bokeh blur |
| [Fast Gaussian Blur](examples/fast_gaussian.tex) | mipmap-accelerated Gaussian approximation |
| [Film Chromatic Aberration](examples/film_chromatic_aberration.tex) | multi-band spectral lateral CA |
| [Film Exponential Blur](examples/film_exponential_blur.tex) | Gaussian-pyramid approximation of exponential decay kernel |
| [Film Grain](examples/film_grain.tex) | density-domain photographic grain synthesis |
| [Film Lens Distortion](examples/film_lens_distortion.tex) | Nuke-style division model |
| [Film Optical Glow](examples/film_optical_glow.tex) | exponential-kernel bloom with highlight isolation |
| [Film Sharpen](examples/film_sharpen.tex) | ratio-based sharpening with edge protection |
| [Film Soften](examples/film_soften.tex) | mipmap-accelerated bilateral filter |
| [Film Vignette](examples/film_vignette.tex) | physically-based photographic vignetting |
| [Fix Pixels](examples/fix_pixels.tex) | sanitize NaN and Inf values in images |
| [Flow & Alligator noise](examples/flow_noise.tex) | compare two exotic noise functions |
| [Frame Blend](examples/frame_blend.tex) | simple 3-frame temporal average |
| [Frame Blend](examples/frame_blend_weighted.tex) | weighted temporal average with adjustable window |
| [Gaussian Blur](examples/gaussian_blur.tex) | Built-in stdlib convenience wrapper |
| [Godrays](examples/godrays.tex) | radial light rays with progressive mipmap softening |
| [Grade](examples/grade.tex) | Nuke-style lift / gamma / gain / offset color grade |
| [Gradient](examples/gradient.tex) | procedural two-color linear gradient |
| [Grain](examples/grain.tex) | simple procedural film grain (training example) |
| [Grayscale](examples/grayscale.tex) | luminance-based desaturation with dual output |
| [Halftone](examples/halftone.tex) | rotated dot-pattern halftone effect |
| [Hue Shift](examples/hue_shift.tex) | rotate the hue of an image |
| [Image Gradient](examples/image_gradient.tex) | visualize the luminance gradient of an image |
| [Invert](examples/invert.tex) | flip all color channels |
| [Kaleidoscope](examples/kaleidoscope.tex) | mirror and rotate UVs for a kaleidoscope effect |
| [Latent blend](examples/latent_blend.tex) | mix two latent tensors |
| [Latent scale](examples/latent_scale.tex) | adjust latent intensity |
| [Lens Distortion](examples/lens_distortion.tex) | barrel/pincushion radial distortion |
| [Simple Lens Distortion](examples/lens_distortion_simple.tex) | AE Optics-Compensation-style radial distortion |
| [Levels](examples/levels.tex) | Photoshop-style input/output level mapping |
| [Luma Keyer](examples/luma_keyer.tex) | extract a mask from image brightness |
| [Luminance Key](examples/luminance_key.tex) | matte extraction from brightness with zone targeting |
| [Marble](examples/marble.tex) | Classic turbulence-driven veined stone |
| [Mask from color](examples/mask_from_color.tex) | extract a mask by color similarity |
| [Matrix Transform](examples/matrix_transform.tex) | mat3 for 2D affine UV transforms |
| [Median Filter](examples/median_filter.tex) | 3x3 median denoise using per-channel arrays |
| [Merge](examples/merge.tex) | combine two images with standard compositing operations |
| [Mipmap Blur](examples/mipmap_blur.tex) | progressive vertical blur using mipmap levels |
| [Motion Detect](examples/motion_detect.tex) | cross-frame difference for motion highlighting |
| [Multi Output](examples/multi_output.tex) | extract R, G, B channels as separate images |
| [Multi Sample](examples/multi_sample.tex) | composite multiple inputs using binding access syntax |
| [Normal Map](examples/normal_map.tex) | generate a tangent-space normal map from a height map |
| [Normalize Mask](examples/normalize_mask.tex) | remap mask values to fill the full 0-1 range |
| [Optical Flow](examples/optical_flow.tex) | Lucas-Kanade motion estimation |
| [Perlin Clouds](examples/perlin_clouds.tex) | procedural cloud pattern using FBM noise |
| [Pixelate](examples/pixelate.tex) | mosaic / block effect |
| [Posterize](examples/posterize.tex) | Quantise pixel values to a limited number of discrete levels |
| [Premultiply](examples/premultiply.tex) | premultiply / unpremultiply alpha round-trip |
| [Radial gradient](examples/radial_gradient.tex) | circular falloff from center |
| [Mandelbrot Fractal](examples/recursive_pattern.tex) | escape-time fractal with cosine palette |
| [Sample Comparison](examples/sample_comparison.tex) | Bilinear vs Cubic vs Lanczos resampling |
| [SDF Shapes](examples/sdf_shapes.tex) | all four signed-distance-field primitives |
| [Sharpen](examples/sharpen.tex) | Laplacian 3x3 kernel |
| [Simplex Terrain](examples/simplex_terrain.tex) | procedural terrain heightfield with simplex noise |
| [Soft Clamp](examples/soft_clamp.tex) | clamp pixel values with smooth rolloff |
| [STMap](examples/stmap.tex) | distort an image using a UV map |
| [String Build](examples/string_build.tex) | construct a sanitized filename from string parameters |
| [String Case](examples/string_case.tex) | normalize text by stripping whitespace and lowercasing |
| [String Format](examples/string_format.tex) | demonstrate string functions and image statistics |
| [Swirl](examples/swirl.tex) | swirl distortion using bicubic sampling |
| [Temporal processing](examples/temporal_functions.tex) | motion-adaptive temporal smoothing |
| [Temporal Median](examples/temporal_median.tex) | median filter across frames |
| [Temporal Ramp](examples/temporal_ramp.tex) | fade in/out over a frame sequence |
| [Ternary Chain](examples/ternary_chain.tex) | multi-level luminance classification |
| [Threshold Mask](examples/threshold_mask.tex) | convert image luminance to a binary mask |
| [Tilt-Shift](examples/tilt_shift.tex) | simulate miniature/diorama effect using mipmap blur |
| [Time Echo](examples/time_echo.tex) | Nuke-style temporal echo effect |
| [Tone Map](examples/tone_map.tex) | ACES filmic curve |
| [2D Transform](examples/transform_2d.tex) | translate, rotate, scale with pivot point |
| [Turbulent Displacement](examples/turbulent_displace.tex) | noise-driven image distortion |
| [Unsharp Mask](examples/unsharp_mask.tex) | classic image sharpening technique |
| [User Function Library](examples/user_function_lib.tex) | composable function definitions |
| [Vec4 Median](examples/vec4_median.tex) | 3x3 spatial median filter using vec4 arrays |
| [Vector Blur](examples/vector_blur.tex) | directional per-pixel motion blur driven by a vector map |
| [Vignette](examples/vignette.tex) | darken image edges with soft circular falloff |
| [Voronoi Cells](examples/voronoi_cells.tex) | Distance-based cellular pattern |
| [While Loop](examples/while_loop.tex) | Newton's method for square root |
| [White Balance](examples/white_balance.tex) | Temperature and Tint adjustment |
| [Wood Grain](examples/wood_grain.tex) | Concentric rings with FBM perturbation |
| [ZDefocus](examples/zdefocus.tex) | depth-aware variable-radius defocus blur |

## Function coverage (soft)

77/143 registered stdlib functions are exercised by an example. The 66 below are not — a nudge to add one, not a build break:

`asin`, `atop`, `bilateral_filter`, `ceil`, `char_at`, `color_burn`, `color_dodge`, `contains`, `cosh`, `count`, `cross`, `degrees`, `determinant`, `dilate`, `endswith`, `erode`, `find`, `fit`, `hard_light`, `hash`, `hash_float`, `hash_int`, `img_mean`, `img_median`, `img_sum`, `inverse`, `join`, `len`, `linear_light`, `linear_to_srgb`, `lstrip`, `matches`, `mix`, `normalize`, `oklab_from_rgb`, `oklab_to_rgb`, `over`, `pad_left`, `pad_right`, `perlin`, `premultiply`, `reflect`, `repeat`, `replace`, `reverse`, `ridged`, `round`, `rstrip`, `sample_frame`, `sign`, `sinh`, `split`, `srgb_to_linear`, `startswith`, `str`, `str_reverse`, `substr`, `tanh`, `to_float`, `to_int`, `trunc`, `under`, `unpremultiply`, `upper`, `vivid_light`, `voronoi`
