from torch import nn

import torch 
import math 

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    # Save original config settings
    use_cache = model.config.use_cache
    model.config.use_cache = False

    batch_size = 1
    n_batches = math.ceil(nsamples / batch_size)

    nlls = []

    # Process in batches
    from tqdm import tqdm
    for batch_idx in tqdm(range(n_batches)):
        # print("Batch", batch_idx, "of", n_batches, flush=True)
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, nsamples)
        current_batch_size = end_idx - start_idx

        # Prepare batch input
        batch_input_ids = torch.stack([
            testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
            for i in range(start_idx, end_idx)
        ]).squeeze(1).to(model.device)

        # Forward pass through the entire model at once
        outputs = model(
            input_ids=batch_input_ids,
            use_cache=False,
            return_dict=True
        )

        # Calculate loss for each item in batch
        for j in range(current_batch_size):
            logits = outputs.logits[j]
            shift_logits = logits[:-1, :].contiguous()
            shift_labels = batch_input_ids[j, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)
        
        # OOM
        del outputs
        torch.cuda.empty_cache()

    # Calculate perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    # Restore original config
    model.config.use_cache = use_cache

    return ppl.item()

