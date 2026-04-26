--!strict

-- Constant(): Set all elements to a constant value in a matrix
-- @param rows: number
-- @param cols: number
-- @param constant?: Default 0
-- @return {{ number }}
return function (rows: number, cols: number, constant: number?): {{ number }}
	local constant = constant or 0

	local m = table.create(rows)
	for i = 1, rows do
		m[i] = table.create(cols, constant)
	end

	return m :: {{ number }}
end