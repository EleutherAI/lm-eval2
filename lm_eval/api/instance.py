import abc


class Instance(abc.ABC):
    """
    A class used to bind together all necessary information and metadata for
    running forward pass of a model on a specific datapoint.

    """

    # all Instance subclasses have an attribute which is the name of the LM() class function they call to get outputs.
    request_type = None

    def __init__(self, doc, arguments=None, doc_idx=None, repeats: int = 1):

        self.doc = doc  # store the document which we're using. this is a dict
        self.doc_idx = doc_idx  # index of the doc within valid/test set
        self.task_name = None
        self._arguments = arguments

        self.repeats = repeats

        # lists containing: 1) the inputs and targets for each, 2) the outputs from the model, 3) the outputs from the model after applying filtering
        self.resps = None
        self.filtered_resps = None

        # TODO: add more info as needed for detailed logging


class LoglikelihoodInstance(Instance):

    request_type = "loglikelihood"

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @property
    def arguments(self):
        """
        Returns (context,target) where `context` is the input and `target` is
        the string to calculate loglikelihood over, conditional on `context` preceding it.
        """
        return self._arguments


class RollingLoglikelihoodInstance(Instance):

    request_type = "loglikelihood_rolling"

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @property
    def arguments(self):
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return (
            self._arguments if isinstance(self._arguments, tuple) else (self.arguments,)
        )


class GenerationInstance(Instance):

    request_type = "greedy_until"

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # TODO: generation/model fwd pass kwargs here and should be passed through arguments as well

    @property
    def arguments(self):
        """
        Returns (string, until) where `string` is the input sequence beginning generation and
        `until` is a string or list of strings corresponding to stop sequences.
        """
        return self._arguments
