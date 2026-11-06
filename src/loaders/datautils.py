def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_lerobot_data(nsamples, seed, policy, pretrained_path,
                     repo_id="lerobot/svla_so100_pickplace", n_test_episodes=5):
    import random
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors

    meta = LeRobotDatasetMetadata(repo_id)

    # ask for a full action chunk per sample (ground truth to compare against)
    chunk = policy.config.chunk_size
    delta_timestamps = {"action": [t / meta.fps for t in range(chunk)]}

    # split by EPISODE, not by frame, so test episodes are never seen in calibration
    episodes = list(range(meta.total_episodes))
    random.seed(seed)
    random.shuffle(episodes)
    test_eps, train_eps = episodes[:n_test_episodes], episodes[n_test_episodes:]

    train = LeRobotDataset(repo_id, episodes=train_eps, delta_timestamps=delta_timestamps)
    test = LeRobotDataset(repo_id, episodes=test_eps, delta_timestamps=delta_timestamps)

    # SmolVLA's own preprocessing: normalization, instruction tokenization, image resizing
    preprocess, postprocess = make_pre_post_processors(policy.config, pretrained_path=pretrained_path)

    def to_batch(sample):
        batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in sample.items()}
        return preprocess(batch)

    idx = random.sample(range(len(train)), nsamples)
    trainloader = [to_batch(train[i]) for i in idx]
    return trainloader, test, to_batch, postprocess