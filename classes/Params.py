class Params():
    """
    Class to hold parameters for the model
    """
    
    def __init__(self,
                 params: dict
                 ) -> None:
        """
        Parameters
        ----------
        params : dict : Dictionary of parameters to be set as attributes
        """
        for key, value in params.items():
            setattr(self, key, value)
