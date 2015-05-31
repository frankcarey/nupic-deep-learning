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

dirpath = "images-duck-n01846331/"
limit = 20

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

results = {}
result_id = 1
for file in img_list do
  local filename = dirpath .. file;
  img = ImgLoader:load(filename);
  if not img then goto continue end
  --itorch.image(img)
  output = net:forward(img)
  pred_a = Predictions:new{output = output}
  pred_a:init()
  pred_a:sort()
  categories = pred_a:getN(10)
  sdr = {}    -- new array

  -- We only have 1000 categories, but make NuPic Happy
  for i=1, 1024 do
    sdr[i] = 0
  end
  for k,c in pairs(categories) do
    if c.confidence > .01 then
      sdr[c.category_id] = 1
      print(c)
    end
  end
  table.insert(results, {id = result_id, SDR = sdr, Class = "duck", file = filename })
  ::continue::
  result_id = result_id + 1
  if result_id > limit then
    goto complete
  end
end
::complete::
print("DONE! ... processing file")
-- Format the results
JSON = (loadfile "JSON.lua")()
local raw_json_text = JSON:encode(results)
local file = io.open("ducks.json", "w")
--print(raw_json_text)
file:write(raw_json_text)
file:close()
print("Output File created.")
