--!strict

return 
	-- Dropout(): Apply dropout to a hidden state
	-- @param h: {{ number }}
	-- @param rate: number
	-- @param training: boolean
	-- @return {{ number }}
	function(h: {{ number }}, rate: number, training: boolean): {{ number }}
		if not training or rate == 0 then
			return h
		end

		local scale = 1 / (1 - rate)

		local result = table.create(#h)
		for i = 1, #h do
			local keep = math.random() > rate and 1 or 0
			result[i] = { h[i][1] * keep * scale }
		end

		return result :: {{ number }}
	end