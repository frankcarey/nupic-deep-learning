local Predictions = { output2 = nil; predictions = {} }

function Predictions:init(net_output)
  something = 'test'
  self.output2 = net_output:float()
  local label = require 'overfeat_label'
  for i=1,self.output2:size()[1] do
    table.insert(self.predictions, {category_id=i, label=label[i], confidence=self.output2[i]})
  end
end

-- output needs to be from output = net:forward(input)
function Predictions:sort()
  table.sort(self.predictions, function(a,b) return a.confidence > b.confidence end)
end

function Predictions:getN(limit)
  local t={};
  for i=1,limit do
    t[i] = self.predictions[i]
  end
  return t
end

return Predictions
