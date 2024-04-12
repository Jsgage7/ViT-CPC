from torch import nn

def pixelCNN(latents) :
# latents: [B, H, W, D]
    cres = latents
    cres_dim = cres . shape[ - 1 ]
    for _ in range ( 5 ) :
        c = Conv2D(output_channels=256 ,
        kernel_shape= ( 1 , 1 ) ) (cres)
        c = ReLU(c)
        c = Conv2D(output_channels=256 ,
        kernel_shape= ( 1 , 3 ) ) (c)
        c = Pad(c , [ [ 0 , 0 ] , [ 1 , 0 ] , [ 0 , 0 ] , [ 0 , 0 ] ] )
        c = Conv2D(output_channels=256 ,
        kernel_shape= ( 2 , 1 ) ,
        type='VALID') (c)
        c = ReLU(c)
        c = Conv2D(output_channels=cres_dim ,
        kernel_shape= ( 1 , 1 ) ) (c)
        cres = cres + c
    cres = ReLU(cres)
    return cres



def CPC(latents , target_dim=64 , emb_scale= 0.1 , steps_to_ignore=2 , steps_to_predict= 3 ):
# latents: [B, H, W, D]
    loss = 0.0
    context = pixelCNN(latents)
    targets = Conv2D(output_channels=target_dim , kernel_shape= ( 1 , 1 ))(latents)
    batch_dim , col_dim , rows = targets . shape [ : -1 ]
    targets = reshape(targets , [ - 1 , target_dim ] )
    for i in range(steps_to_ignore , steps_to_predict) :
        col_dim_i = col_dim - i - 1
        total_elements = batch_dim * col_dim_i * rows
        preds_i = Conv2D(output_channels=target_dim ,
        kernel_shape= ( 1 , 1 ) ) (context)
        preds_i = preds_i [ : , : - (i+ 1 ) , : , : ] * emb_scale
        preds_i = reshape(preds_i , [ - 1 , target_dim ] )
        logits = matmul(preds_i , targets , transp_b=True)
        b = range(total_elements) / (col_dim_i * rows)
        col = range(total_elements) % (col_dim_i * rows)
        labels = b * col_dim * rows + (i+ 1 ) * rows + col
        loss += cross_entropy_with_logits(logits , labels)
    return loss