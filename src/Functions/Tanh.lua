--!strict

return {
	-- Tanh(): Tanh activation function
	-- @param x: Input value
	-- @return number: Tanh of x
	Activation = function(x: number): number
		return math.tanh(x)
	end,

	-- TanhDerivative(): Tanh derivative
	-- @param x: Input value
	-- @return number: Derivative value
	Derivative = function(x: number): number
		local t = math.tanh(x)

		return 1 - t * t
	end
}