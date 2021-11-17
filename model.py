class BaselineUnet4(nn.Module):
    '''
    This class implements a simple UNet as in Ronneberger et al, but deprived of one encoding/decoding block.
    This is done in order to handle the flow of information in the bottleneck, associated to the smaller 128x128 shaped patches used in our paper.
        
        Inputs: in_channel, out_channel
        * **in_channels**: number of channels in the input tensor (i.e. 2 for SAR images)
        * **out_channel**: number of output classes
    '''
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        """
        This function creates one contracting block
        """ 
        block = torch.nn.Sequential(        
       
        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,padding=1),  
        torch.nn.ReLU(),   
        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels,padding=1),  
        torch.nn.ReLU(),    
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """ 
        block = torch.nn.Sequential(  
        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel,padding=1),     
        torch.nn.ReLU(), 
        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel,padding=1),     
        torch.nn.ReLU(),  
        torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1,output_padding=1),  
        )
        return  block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """ 
        block = torch.nn.Sequential( 
        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel,padding=1),       
        torch.nn.ReLU(),
        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel,padding=1),   
        torch.nn.ReLU(), 
        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1) 
        )
        return  block

    def __init__(self, in_channel, out_channel):
        super(BaselineUnet4, self).__init__() 
        #Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(64,128)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)  
        self.conv_encode3 = self.contracting_block(128,256)
        self.conv_maxpool3= torch.nn.MaxPool2d(kernel_size=2)  
        # Bottleneck
        self.bottleneck = torch.nn.Sequential( 
        torch.nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512,padding=1), 
        torch.nn.ReLU(),  
        torch.nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512,padding=1), 
        torch.nn.ReLU(),   
        torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2,padding=1,output_padding=1),      
        )
        #Decode  
        self.conv_decode3 = self.expansive_block(512, 256,128)
        self.conv_decode2 = self.expansive_block(256, 128,64)
        self.final_layer = self.final_block(128, 64, out_channel)

    def concat(self, upsampled, bypass):

        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2) 
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3) 
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3) 
        # Decode    
        decode_block3 = self.concat(bottleneck1, encode_block3)      
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.concat(cat_layer2, encode_block2)      
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.concat(cat_layer1, encode_block1)
        final_layer = self.final_layer(decode_block1)
        return  final_layer
