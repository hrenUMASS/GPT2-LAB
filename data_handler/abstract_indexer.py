class AbstractIndexer:

    def __init__(self, **kwargs):
        self.eval = None

    def get_dataset(self, nth_loader, prototype=None, tokenizer=None, dataset_type=None, batch_len=100, data=None):
        pass

    def get_eval(self, prototype=None, tokenizer=None, dataset_type=None, eval_len=10, data=None):
        pass
