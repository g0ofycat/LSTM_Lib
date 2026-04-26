--!strict

-- get_max(): Get the maximum value from an array of numbers
-- @param input: Array of numbers
-- @return: Maximum value in the array
local function get_max(input: { number }): number
	local max = input[1]

	for i = 2, #input do
		if input[i] <= max then continue end
		max = input[i]
	end

	return max
end

return
	-- Softmax(): Softmax activation function
	-- @param inputs: Array of input values
	-- @return { number }: Array of probabilities
	function (inputs: { number }): { number }
		local max = get_max(inputs)

		local exp_sum = 0
		local outputs = {}

		for i = 1, #inputs do
			outputs[i] = math.exp(inputs[i] - max)
			exp_sum += outputs[i]
		end

		for i = 1, #outputs do
			outputs[i] /= exp_sum
		end

		return outputs
	end