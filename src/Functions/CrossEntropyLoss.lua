--!strict

return
	-- CrossEntropyLoss(): Cross Entropy Loss
	-- @param Outputs: Predicted values
	-- @param Expected: Expected values
	-- @return number: The average error
	function (Outputs: { number }, Expected: { number }): number
		local sum = 0

		for i = 1, #Outputs do
			local pred = math.clamp(Outputs[i], 1e-7, 1 - 1e-7)

			sum += -Expected[i] * math.log(pred)
		end

		return sum / #Outputs
	end