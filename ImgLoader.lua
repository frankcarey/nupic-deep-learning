local  ImgLoader = {}
require "image"
require "paths"

function ImgLoader:load(filepath)
  local img
  local img_raw


  -- Use cuda if out global was set.
  cuda = cuda or false
  if cuda then net:cuda() end

  -- load and preprocess image
  local dim
  if network == 'small' then    dim = 231
  elseif network == 'big' then  dim = 221 end


  if filepath  == 'lena' then
    img_raw = image.lena():mul(255)
    print("loading image.lena()")

  elseif not paths.filep(filepath) then
    print("File not found: " .. filepath)
    return nil
  else
    img_raw = image.load(filepath):mul(255)
  end

  local rh = img_raw:size(2)
  local rw = img_raw:size(3)
  if rh < rw then
     rw = math.floor(rw / rh * dim)
     rh = dim
  else
     rh = math.floor(rh / rw * dim)
     rw = dim
  end
  local img_scale = image.scale(img_raw, rw, rh)

  local offsetx = 1
  local offsety = 1
  if rh < rw then
     offsetx = offsetx + math.floor((rw-dim)/2)
  else
     offsety = offsety + math.floor((rh-dim)/2)
  end
  img = img_scale[{{},{offsety,offsety+dim-1},{offsetx,offsetx+dim-1}}]:floor()

  img:add(-118.380948):div(61.896913)  -- fixed distn ~ N(118.380948, 61.896913^2)

  return img
end

return ImgLoader
