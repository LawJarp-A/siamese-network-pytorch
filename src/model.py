class SiameseNet(pl.LightningModule):
    def __init__(self, model, ptim=None, scheduler=None):
        super().__init__()
        self.model = model
        # Setup optim
        self.optim = optim if (optim != None) else torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # Set up scheduler
        self.scheduler = scheduler
        # Set up Loss function
        self.loss_fn = nn.CosineEmbeddingLoss(margin=0)

    # Define the standard output for the model
    def forward(self, images):
        image1, image2 = images
        features1 = self.model(image1)
        features2 = self.model(image2)
        return (features1, features2)    

    # Used to get embedding for a single image
    def forward_once(self, image):
        return self.model(image)
    

    def configure_optimizers(self):
        optimizer = self.optim
        lr_scheduler = self.scheduler
        
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        f1, f2 = self(x)
        loss = self.loss_fn(f1,f2,y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        f1, f2 = self(x)
        val_loss = self.loss_fn(f1,f2,y)
        self.log("val_loss", val_loss)
        # return self.cos(f1, f2)
        # return F.pairwise_distance(f1,f2, keepdim = True)