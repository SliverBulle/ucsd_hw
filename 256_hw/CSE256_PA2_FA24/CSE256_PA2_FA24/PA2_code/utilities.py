import matplotlib.pyplot as plt
import torch

class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size):
        self.model.eval()
        with torch.no_grad():
            wordids = self.tokenizer.encode(sentence)

            padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
            input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0).to(next(self.model.parameters()).device)
            # input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)

            print("Input tensor shape:", input_tensor.shape)

            logits, attn_maps = self.model(input_tensor)  # get decoder output and attention maps

        
            print("Number of attention maps:", len(attn_maps))


            layer_index = 0  # Layer 1 (index starts from 0)
            head_index = 0   # Head 1 (index starts from 0)

            attn_map = attn_maps[layer_index]  # get attention map of layer 1
            att_map = attn_map.squeeze(0).detach().cpu().numpy()  # shape: (n_head, T, T)

            single_head_att_map = att_map[head_index]  # get attention map of head 1 (T, T)


            total_prob_over_rows = single_head_att_map.sum(axis=1)  # sum
            if (total_prob_over_rows < 0.99).any() or (total_prob_over_rows > 1.01).any():
                print(f"Failed normalization test: probabilities do not sum to 1.0 over rows for Layer {layer_index + 1}, Head {head_index + 1}")
                print("Total probability over rows:", total_prob_over_rows)
            else:
                print(f"All attention rows are correctly normalized for Layer {layer_index + 1}, Head {head_index + 1}.")

            fig, ax = plt.subplots()
            cax = ax.imshow(single_head_att_map, cmap='hot', interpolation='nearest')
            ax.xaxis.tick_top()
            fig.colorbar(cax, ax=ax)
            plt.title(f"Attention Map - Layer {layer_index + 1}, Head {head_index + 1}")

            import time
            current_time = time.strftime("%Y-%m-%d_%H-%M-%S")   
            plt.savefig(f"plot/attention_map_layer{layer_index + 1}_head{head_index + 1}_{current_time}.png")

                


