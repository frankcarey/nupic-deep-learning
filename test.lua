require 'nn'
require 'image'

cuda = false;
net = require "create_network"
output = Tensor
network = 'small'

-- PUT IMAGE STUFF BACK
--
ImgLoader = require 'ImgLoader'
Predictions = require 'predictions'

dirpath = "images/"

if not dirpath or not paths.dirp(dirpath) then
  print ("directory " .. dirpath or "?" .. " isn't set or doesn't exist, using test images.");
  dirpath = ""
  img_list = ImgLoader:list_iter({
    "lena",
    "bee.jpg"
  })
else
  -- Iternate over all images in a path
  img_list = paths.files(dirpath)
end

for file in img_list do
  img = ImgLoader:load(dirpath .. file);
  if not img then goto continue end
  --itorch.image(img)
  output = net:forward(img)
  pred_a = Predictions:new{output = output}
  pred_a:init()
  pred_a:sort()
  print(pred_a:getN(3))
  ::continue::
end
