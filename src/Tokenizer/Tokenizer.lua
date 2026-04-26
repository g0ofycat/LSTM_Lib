--!strict

local Tokenizer = {}
Tokenizer.__index = Tokenizer

--====================
-- // TYPES
--====================

local Types = require("./Utility/Types")
export type Tokenizer_constructor_type = typeof(setmetatable({} :: Types.TokenizerType, Tokenizer))

--====================
-- // CONSTRUCTOR
--====================

-- new(): Create new tokenizer
-- @param vocab: Vocabulary
-- @return Tokenizer_constructor_type
function Tokenizer.new(vocab: { string }): Tokenizer_constructor_type
	local stoi = {}
	for i, token in vocab do
		stoi[token] = i
	end

	return setmetatable({ stoi = stoi, itos = vocab }, Tokenizer)
end

--====================
-- // METHODS
--====================

-- encode(): Split text and map to indices (1-based)
-- @param text: Text to encode
-- @return { number }
function Tokenizer:encode(text: string): { number }
	local unk = self.stoi["<unk>"]
	local indices = {}

    for word in text:lower():gmatch("%S+") do
        indices[#indices + 1] = self.stoi[word] or unk
    end

    return indices
end

-- decode(): Map indices back to words
-- @param indices: { number }
-- @return string
function Tokenizer:decode(indices: { number }): string
	local words = {}
	for i, idx in indices do
		words[i] = self.itos[idx] or "<unk>"
	end

	return table.concat(words, " ")
end

return Tokenizer