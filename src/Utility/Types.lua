--!strict

export type LSTM_Cache = {
	inputs: {{{ number }}},

	h: {{{ number }}},
	c: {{{ number }}},
	f: {{{ number }}},
	i: {{{ number }}},
	g: {{{ number }}},
	o: {{{ number }}},
	y: {{{ number }}},

	h_prev: {{{ number }}},
	c_prev: {{{ number }}}
}

export type LSTM_Data = {
	input_neurons: number,
	hidden_neurons: number,
	output_neurons: number,
	learning_rate: number,
	dropout_rate: number,

	weights: { [number]: { [number]: { [number]: number } } }, -- // [layer][from][to]

	bf: {{ number }},
	bi: {{ number }},
	bg: {{ number }},
	bo: {{ number }},
	by: {{ number }},

	h: {{ number }},
	c: {{ number }},

	embed_weight: { [number]: { [number]: number } },

	cache: LSTM_Cache
}

return nil