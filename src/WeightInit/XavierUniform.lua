--!strict

-- XavierUniform(): Xavier Weight Initialization (In Place)
-- @param weights: { [number]: { [number]: { [number]: number } } }
-- @param fromSize: number
-- @param toSize: number
-- @param layer: number
-- @param random_generator: Random
return function (weights: { [number]: { [number]: { [number]: number } } }, fromSize: number, toSize: number, layer: number, random_generator: Random): ()
	weights[layer] = {}

	local scale = math.sqrt(6 / (fromSize + toSize))

	for from = 1, fromSize do
		weights[layer][from] = {}

		for to = 1, toSize do
			weights[layer][from][to] = random_generator:NextNumber(-scale, scale)
		end
	end
end