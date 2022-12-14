# %%
my_device = "cuda" if torch.cuda.is_available() else "cpu"    
model = model.to(my_device)    
losses = []
time_elapsed = []
epochs = []
# t0 = time.time()    
for epoch in range(num_epochs):        
    total_loss = 0.0
    batch_count = 0        
    for batch in train_loader:            
            optF.zero_grad()
            batch = batch.to(my_device)
            pred = model(batch, batch.ndata["feat"].to(my_device))
            loss = loss_fn(pred, batch.ndata["label"].to(my_device))
            loss.backward()
            optimizer.step()            
            total_loss += loss.detach()
            batch_count += 1        
            mean_loss = total_loss / batch_count
