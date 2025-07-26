import wandb

# define a singleton class for wandb

class WandbSingleton:

    def __init__(self):
        self.initialized = False

    def add_config(self, args):
        """
        Add configuration to the wandb run.
        
        Args:
            config (dict): Configuration dictionary to log.
        """
        if self.initialized:
            raise RuntimeError("WandbSingleton is already initialized. Call finish before reinitializing.")
        

        if args.use_wandb:
            self.active = True    

            # TODO format a wandb run name
            self.run = wandb.init(
                project="openfgl",
                entity="lourenst-freelance",
                config=args.__dict__,  # Convert Namespace to dict
            )
            self.run.config.update(args)
        
        else:
            self.active = False
            self.run = None
        
        self.initialized = True
    
    def log(self, data: dict):
        """
        Log data to the wandb run.
        
        Args:
            data (dict): Data to log.
        """
        if not self.initialized:
            raise RuntimeError("WandbSingleton not initialized. Call add_config first.")

        if self.active and self.run is not None:
            self.run.log(data)

    def finish(self):
        """
        Finish the wandb run.
        """
        if self.active and self.run is not None:
            self.run.finish()
    
# Create a global instance of the WandbSingleton
wandb_run = WandbSingleton()