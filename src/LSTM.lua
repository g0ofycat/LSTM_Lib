--!strict

local LSTM = {}
LSTM.__index = LSTM

--================
-- // MODULES
--================

local Config = require("./Utility/Config")

local Matrix = require("./Utility/Matrix")

local Dropout = require("./Functions/Dropout")

local XavierInit = require("./WeightInit/XavierUniform")
local Constant = require("./BiasInit/Constant")

local CrossEntropyLoss = require("./Functions/CrossEntropyLoss")
local Softmax = require("./Functions/Softmax")

local Sigmoid = require("./Functions/Sigmoid")
local Tanh = require("./Functions/Tanh")

--================
-- // TYPES
--================

local Types = require("./Utility/Types")
export type LSTM_constructor_type = typeof(setmetatable({} :: Types.LSTM_Data, LSTM))

--====================
-- // CONSTRUCTOR
--====================

-- new(): Construct a new LSTM
-- @param i_neurons: Number of input neurons
-- @param h_neurons: Number of hidden neurons
-- @param o_neurons: Number of output neurons
-- @param l_rate: Learning rate
-- @param d_rate: Dropout rate
-- @return: A new LSTM
function LSTM.new(i_neurons: number, h_neurons: number, o_neurons: number, l_rate: number, d_rate: number): LSTM_constructor_type
	local rng = Random.new()
	local w = {}

	XavierInit(w, h_neurons, i_neurons, 1, rng)
	XavierInit(w, h_neurons, i_neurons, 3, rng)
	XavierInit(w, h_neurons, i_neurons, 5, rng)
	XavierInit(w, h_neurons, i_neurons, 7, rng)

	XavierInit(w, h_neurons, h_neurons, 2, rng)
	XavierInit(w, h_neurons, h_neurons, 4, rng)
	XavierInit(w, h_neurons, h_neurons, 6, rng)
	XavierInit(w, h_neurons, h_neurons, 8, rng)

	XavierInit(w, o_neurons, h_neurons, 9, rng)

	return setmetatable({
		input_neurons = i_neurons,
		hidden_neurons = h_neurons,
		output_neurons = o_neurons,
		learning_rate = l_rate,
		dropout_rate = d_rate,

		weights = w,

		bf = Constant(h_neurons, 1),
		bi = Constant(h_neurons, 1),
		bg = Constant(h_neurons, 1),
		bo = Constant(h_neurons, 1),
		by = Constant(o_neurons, 1),

		h = Constant(h_neurons, 1),
		c = Constant(h_neurons, 1),

		embed_weight = {},

		cache = {
			inputs = {},

			h = {},
			c = {},
			f = {},
			i = {},
			g = {},
			o = {},
			y = {},

			h_prev = {},
			c_prev = {}
		}
	}, LSTM)
end

--====================
-- // PROPAGATION METHODS
--====================

-- forward_step(): Perform a forward pass through the LSTM
-- @param token_idx: Index of the input token
-- @param training: Enable dropout
-- @return: {{ number }}
function LSTM:forward_step(token_idx: number, training: boolean): {{ number }}
	local self = self :: LSTM_constructor_type
	local cache = self.cache

	local input = {}
    local row = self.embed_weight[token_idx]
    for i = 1, #row do
        input[i] = { row[i] }
    end

	local function gate(W: number, U: number, b: {{ number }}, fn: (number) -> number): {{ number }}
		return Matrix.apply(
			Matrix.eadd(
				Matrix.eadd(Matrix.matmul(self.weights[W], input), Matrix.matmul(self.weights[U], self.h)),
				b
			),
			fn
		)
	end

	local f = gate(1, 2, self.bf, Sigmoid.Activation)
	local i = gate(3, 4, self.bi, Sigmoid.Activation)
	local g = gate(5, 6, self.bg, Tanh.Activation)
	local o = gate(7, 8, self.bo, Sigmoid.Activation)

	local h_prev = self.h
	local c_prev = self.c

	self.c = Matrix.eadd(Matrix.emul(f, self.c), Matrix.emul(i, g))
	self.h = Dropout(Matrix.emul(o, Matrix.apply(self.c, Tanh.Activation)), self.dropout_rate, training)

	table.insert(cache.inputs, input)

	table.insert(cache.f, f)
	table.insert(cache.i, i)
	table.insert(cache.g, g)
	table.insert(cache.o, o)

	table.insert(cache.h, self.h)
	table.insert(cache.c, self.c)

	table.insert(cache.h_prev, h_prev)
	table.insert(cache.c_prev, c_prev)

	local y = Matrix.softmax(
		Matrix.eadd(Matrix.matmul(self.weights[9], self.h), self.by),
		Softmax
	)

	table.insert(cache.y, y)

	return y
end

-- backward(): Backpropagate through time and update weights
-- @param targets: Array of target indices
-- @return: number (average loss)
function LSTM:backward(targets: { number }): number
	local self = self :: LSTM_constructor_type

	local w = self.weights
	local cache = self.cache
	local lr = self.learning_rate
	local T = #cache.inputs

	local dWf = Constant(#w[1], #w[1][1])
	local dUf = Constant(#w[2], #w[2][1])
	local dbf = Constant(self.hidden_neurons, 1)
	local dWi = Constant(#w[3], #w[3][1])
	local dUi = Constant(#w[4], #w[4][1])
	local dbi = Constant(self.hidden_neurons, 1)
	local dWg = Constant(#w[5], #w[5][1])
	local dUg = Constant(#w[6], #w[6][1])
	local dbg = Constant(self.hidden_neurons, 1)
	local dWo = Constant(#w[7], #w[7][1])
	local dUo = Constant(#w[8], #w[8][1])
	local dbo = Constant(self.hidden_neurons, 1)
	local dWy = Constant(#w[9], #w[9][1])
	local dby = Constant(self.output_neurons, 1)

	local dh_next = Constant(self.hidden_neurons, 1)
	local dc_next = Constant(self.hidden_neurons, 1)
	local total_loss = 0

	for t = T, 1, -1 do
		local ht = cache.h[t]
		local ct = cache.c[t]
		local ft = cache.f[t]
		local it = cache.i[t]
		local gt = cache.g[t]
		local ot = cache.o[t]
		local yt = cache.y[t]

		local h_prev = cache.h_prev[t]
		local c_prev = cache.c_prev[t]

		local x = cache.inputs[t]

		local y_flat = table.create(#yt) :: { number }
		for row = 1, #yt do y_flat[row] = yt[row][1] end

		local t_flat = table.create(self.output_neurons, 0) :: { number }
		t_flat[targets[t]] = 1

		total_loss += CrossEntropyLoss(y_flat, t_flat)

		local dy = Constant(self.output_neurons, 1)
		for row = 1, self.output_neurons do
			dy[row][1] = y_flat[row] - t_flat[row]
		end

		dWy = Matrix.eadd(dWy, Matrix.matmul(dy, Matrix.transpose(ht)))
		dby = Matrix.eadd(dby, dy)

		local dh = Matrix.eadd(Matrix.matmul(Matrix.transpose(w[9]), dy), dh_next)

		local tanh_c = Matrix.apply(ct, Tanh.Activation)

		local do_ = Matrix.emul(dh, tanh_c)

		local dc = Matrix.eadd(
			Matrix.emul(dh, Matrix.emul(ot, Matrix.apply(tanh_c, function(v) return 1 - v * v end))),
			dc_next
		)

		local df = Matrix.emul(dc, c_prev)
		local di = Matrix.emul(dc, gt)
		local dg = Matrix.emul(dc, it)

		local dff = Matrix.emul(df, Matrix.apply(ft, function(v) return v * (1 - v) end))
		local dif = Matrix.emul(di, Matrix.apply(it, function(v) return v * (1 - v) end))
		local dgf = Matrix.emul(dg, Matrix.apply(gt, function(v) return 1 - v * v end))
		local dof = Matrix.emul(do_, Matrix.apply(ot, function(v) return v * (1 - v) end))

		local xT = Matrix.transpose(x)
		local hpT = Matrix.transpose(h_prev)

		dWf = Matrix.eadd(dWf, Matrix.matmul(dff, xT))
		dUf = Matrix.eadd(dUf, Matrix.matmul(dff, hpT))
		dbf = Matrix.eadd(dbf, dff)
		dWi = Matrix.eadd(dWi, Matrix.matmul(dif, xT))
		dUi = Matrix.eadd(dUi, Matrix.matmul(dif, hpT))
		dbi = Matrix.eadd(dbi, dif)
		dWg = Matrix.eadd(dWg, Matrix.matmul(dgf, xT))
		dUg = Matrix.eadd(dUg, Matrix.matmul(dgf, hpT))
		dbg = Matrix.eadd(dbg, dgf)
		dWo = Matrix.eadd(dWo, Matrix.matmul(dof, xT))
		dUo = Matrix.eadd(dUo, Matrix.matmul(dof, hpT))
		dbo = Matrix.eadd(dbo, dof)

		dh_next = Matrix.eadd(
			Matrix.eadd(Matrix.matmul(Matrix.transpose(w[2]), dff), Matrix.matmul(Matrix.transpose(w[4]), dif)),
			Matrix.eadd(Matrix.matmul(Matrix.transpose(w[6]), dgf), Matrix.matmul(Matrix.transpose(w[8]), dof))
		)

		dc_next = Matrix.emul(dc, ft)
	end

	local function update(W: {{ number }}, dW: {{ number }}): {{ number }}
		return Matrix.eadd(W, Matrix.scale(Matrix.apply(dW, function(v)
			return math.clamp(v, -Config.PROPAGATION.grad_clip, Config.PROPAGATION.grad_clip)
		end), -lr))
	end

	w[1] = update(w[1], dWf)
	w[2] = update(w[2], dUf)
	w[3] = update(w[3], dWi)
	w[4] = update(w[4], dUi)
	w[5] = update(w[5], dWg)
	w[6] = update(w[6], dUg)
	w[7] = update(w[7], dWo)
	w[8] = update(w[8], dUo)
	w[9] = update(w[9], dWy)

	self.bf = update(self.bf, dbf)
	self.bi = update(self.bi, dbi)
	self.bg = update(self.bg, dbg)
	self.bo = update(self.bo, dbo)
	self.by = update(self.by, dby)

	self:reset_state()

	return total_loss / T
end

--====================
-- // UTILITY METHODS
--====================

-- reset_state(): Reset long and short term states
function LSTM:reset_state(): ()
	local self = self :: LSTM_constructor_type
	self.h = Constant(self.hidden_neurons, 1)
	self.c = Constant(self.hidden_neurons, 1)
	self.cache = { inputs={}, h={}, c={}, f={}, i={}, g={}, o={}, y={}, h_prev={}, c_prev={} }
end

-- export_data(): Export LSTM Data
-- @return: Types.LSTM_Data
function LSTM:export_data(): Types.LSTM_Data
	local self = self :: LSTM_constructor_type

	return {
		input_neurons = self.input_neurons,
		hidden_neurons = self.hidden_neurons,
		output_neurons = self.output_neurons,
		learning_rate = self.learning_rate,
		dropout_rate = self.dropout_rate,

		weights = self.weights,

		bf = self.bf,
		bi = self.bi,
		bg = self.bg,
		bo = self.bo,
		by = self.by,

		h = self.h,
		c = self.c,

		embed_weight = self.embed_weight,

		cache = self.cache
	}
end

--====================
-- // STATIC METHODS
--====================

-- from_data(): Load LSTM from exported data
-- @param data: Types.LSTM_Data
-- @return LSTM_constructor_type
function LSTM.from_data(data: Types.LSTM_Data): LSTM_constructor_type
	return setmetatable({
		input_neurons = data.input_neurons,
		hidden_neurons = data.hidden_neurons,
		output_neurons = data.output_neurons,
		learning_rate = data.learning_rate,
		dropout_rate = data.dropout_rate,

		weights = data.weights,

		bf = data.bf,
		bi = data.bi,
		bg = data.bg,
		bo = data.bo,
		by = data.by,

		h = data.h,
		c = data.c,

		embed_weight = data.embed_weight,

		cache = data.cache
	}, LSTM)
end

return LSTM