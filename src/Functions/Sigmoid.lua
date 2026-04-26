--!strict

return {
	-- Sigmoid(): Sigmoid activation function
	-- @param x: Input value
	-- @return number: Output between 0 and 1
	Activation = function(x: number): number
		if x >= 0 then
			return 1 / (1 + math.exp(-x))
		else
			return math.exp(x) / (1 + math.exp(x))
		end
	end,

	-- SigmoidDerivative(): Sigmoid derivative
	-- @param x: Input value
	-- @return number: Derivative value
	Derivative = function(x: number): number
		local s: number

		if x >= 0 then
			s = 1 / (1 + math.exp(-x))
		else
			s = math.exp(x) / (1 + math.exp(x))
		end

		return s * (1 - s)
	end
}