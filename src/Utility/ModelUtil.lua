--!strict

local ModelUtil = {}

--================
-- // MODULES
--================

local LSTM = require("../LSTM")

--================
-- // PUBLIC API
--================

-- TrainModel(): Train model
-- @param model: LSTM
-- @param epochs: Number of epochs
-- @param dataset: Matrix of sequences
-- @param batch_size: Number of sequences per batch (defaults to full)
function ModelUtil.TrainModel(
	model: LSTM.LSTM_constructor_type,
	epochs: number,
	dataset: {{ number }},
	batch_size: number?
): ()
	local bsz = batch_size or #dataset

	for curr_epoch = 1, epochs do
		local epoch_loss = 0
		local num_batches = 0

		for batch_start = 1, #dataset, bsz do
			local batch_loss = 0
			local batch_end = math.min(batch_start + bsz - 1, #dataset)

			for seq_idx = batch_start, batch_end do
				local sequence = dataset[seq_idx]

				for t = 1, #sequence - 1 do
					model:forward_step(sequence[t], true)
				end

				local targets = {}
				for t = 2, #sequence do
					targets[t - 1] = sequence[t]
				end

				batch_loss += model:backward(targets)
			end

			epoch_loss += batch_loss / (batch_end - batch_start + 1)
			num_batches += 1
		end

		print(`[Epoch {curr_epoch}] - avg loss: {epoch_loss / num_batches}`)
	end
end

-- Predict(): Run inference on a sequence
-- @param model: LSTM
-- @param indices: Array of indices
-- @return {{ number }}: Softmax logits from final step
function ModelUtil.Predict(model: LSTM.LSTM_constructor_type, indices: { number }): {{ number }}
	model:reset_state()

	local output = {} :: {{ number }}
	for _, idx in indices do
		output = model:forward_step(idx, false)
	end

	return output
end

return ModelUtil