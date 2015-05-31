local Predictions = {
  output = {}
  predictions = {}
}

function Predictions:init(output)
  self.output = output:float()
end

-- output needs to be from output = net:forward(input)
function Predictions:get()
  local label = require 'overfeat_label'
  for i=1,self.output:size()[1] do
    table.insert(self.predictions, {category_id=i, label=label[i], confidence=self.output[i]})
  end
  return table.sort(self.predictions, function(a,b) return a.confidence > b.confidence end)
end

return Predictions
