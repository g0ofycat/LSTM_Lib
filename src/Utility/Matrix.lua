--!strict

local Matrix = {}

--================
-- // PUBLIC API
--================

-- matmul(): Matmul
-- @param A: {{ number }}
-- @param B: {{ number }}
-- @return {{ number }}
function Matrix.matmul(A: {{ number }}, B: {{ number }}): {{ number }}
	assert(A[1] and B[1] and #A[1] == #B, "matmul() shape")

	local rows, cols, inner = #A, #B[1], #B
	local result = table.create(rows)

	for i = 1, rows do
		local row = table.create(cols, 0)
		for j = 1, cols do
			local sum = 0
			for k = 1, inner do
				sum += A[i][k] * B[k][j]
			end

			row[j] = sum
		end

		result[i] = row
	end

	return result :: {{ number }}
end

-- eadd(): Elementwise add
-- @param A: {{ number }}
-- @param B: {{ number }}
-- @return {{ number }}
function Matrix.eadd(A: {{ number }}, B: {{ number }}): {{ number }}
	assert(#A == #B and A[1] and B[1] and #A[1] == #B[1], "eadd() shape")

	local result = table.create(#A)

	for i = 1, #A do
		local row = table.create(#A[i])
		for j = 1, #A[i] do
			row[j] = A[i][j] + B[i][j]
		end

		result[i] = row
	end

	return result :: {{ number }}
end

-- emul(): Elementwise multiply
-- @param A: {{ number }}
-- @param B: {{ number }}
-- @return {{ number }}
function Matrix.emul(A: {{ number }}, B: {{ number }}): {{ number }}
	assert(#A == #B and A[1] and B[1] and #A[1] == #B[1], "emul() shape")

	local result = table.create(#A)

	for i = 1, #A do
		local row = table.create(#A[i])
		for j = 1, #A[i] do
			row[j] = A[i][j] * B[i][j]
		end

		result[i] = row
	end

	return result :: {{ number }}
end

-- apply(): Apply a scalar function to every element
-- @param A: {{ number }}
-- @param fn: (number) -> number
-- @return {{ number }}
function Matrix.apply(A: {{ number }}, fn: (number) -> number): {{ number }}
	local result = table.create(#A)

	for i = 1, #A do
		local row = table.create(#A[i])
		for j = 1, #A[i] do
			row[j] = fn(A[i][j])
		end

		result[i] = row
	end

	return result :: {{ number }}
end

-- transpose(): Transpose a matrix
-- @param A: {{ number }}
-- @return {{ number }}
function Matrix.transpose(A: {{ number }}): {{ number }}
	assert(A[1], "transpose() empty")

	local rows, cols = #A, #A[1]
	local result = table.create(cols)

	for j = 1, cols do
		local row = table.create(rows)
		for i = 1, rows do
			row[i] = A[i][j]
		end

		result[j] = row
	end

	return result :: {{ number }}
end

-- scale(): Multiply every element by a scalar
-- @param A: {{ number }}
-- @param s: number
-- @return {{ number }}
function Matrix.scale(A: {{ number }}, s: number): {{ number }}
	return Matrix.apply(A, function(x) return x * s end)
end

-- softmax(): Softmax over a column vector
-- @param A: {{ number }}
-- @param fn: Functions.Softmax
-- @return {{ number }}
function Matrix.softmax(A: {{ number }}, fn: ({ number }) -> { number }): {{ number }}
	assert(A[1] and A[1][1], "softmax() not column")

	local flat = table.create(#A) :: { number }
	for i = 1, #A do
		flat[i] = A[i][1]
	end

	local result = fn(flat)
	local out = table.create(#result)
	for i = 1, #result do
		out[i] = { result[i] }
	end

	return out :: {{ number }}
end

-- embed(): One hot encode weight matrix
-- @param W: {{ number }}
-- @param one_hot_input: {{ number }}
-- @return {{ number }}
function Matrix.embed(W: {{ number }}, one_hot_input: {{ number }}): {{ number }}
	local idx = 1
	for i, row in one_hot_input do
		if row[1] == 1 then idx = i break end
	end

	local out = table.create(#W)
	for i, row in W do
		out[i] = { row[idx] }
	end

	return out :: {{ number }}
end

return Matrix