model = Alexnet() # model name
num_epochs = 40
learning_rate = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.BCELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.01)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
train_losses = []
valid_losses = []
valid_acc = []
train_acc = []

for epoch in range(1, num_epochs+1):
    model.train()
    
    train_loss = 0
    train_correct_pred = 0
    train_total = 0
    
    valid_loss = 0
    valid_correct_pred = 0
    valid_total = 0
    
    since = time.time() 
    for i, (timages, tlabels) in enumerate(train_loader):
        timages = timages.to(device)
        tlabels = tlabels.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        toutputs = model(timages)
        _, tpred = torch.max(toutputs, 1)
        train_total += tlabels.size(0)
        train_correct_pred += (tpred==tlabels).sum().item()
        
        #print(f"\noutput: {tpred},\tlabels: {tlabels}")
        tloss = criterion(toutputs,tlabels)
        
        # Backward and optimize
        tloss.backward()
        optimizer.step()
        train_loss += tloss.item()*timages.size(0)
    
    t_acc = 100.0 * train_correct_pred/train_total
    
    time_elapsed = time.time() - since
    print('Epoch:{:d}/{:d} Training complete in {:.0f}m {:.0f}s'.format(epoch,num_epochs,time_elapsed // 60, time_elapsed % 60))
    train_acc.append(t_acc)
    
    model.eval()
    for i, (vimages, vlabels) in enumerate(test_loader):
        vimages = vimages.to(device)
        vlabels = vlabels.to(device)
        voutputs = model(vimages)
        _, vpred = torch.max(voutputs, 1)
        valid_total += vlabels.size(0)
        valid_correct_pred += (vpred==vlabels).sum().item()
        #print(f"{i}. output- >{outputs}, label {labels}")
        vloss = criterion(voutputs, vlabels)
        valid_loss += vloss.item()*vimages.size(0)
    
    v_acc = 100.0 * valid_correct_pred/valid_total
    valid_acc.append(v_acc)
    
    # calculate average losses
    train_loss = train_loss/len(train_loader)
    valid_loss = valid_loss/len(test_loader)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss) 
    #print(f'train_acc ={train_correct_pred}, valid_acc ={valid_correct_pred}')   
      #if (i-0)%30 == 0: print(f"Epoch [{epoch+1}/{num_epochs}], Step = [{i+1}/{n_total_steps}], Loss = {loss.item():.4f}")
    
    print(f'Epoch: {epoch}/{num_epochs} \tTraining Loss: {train_loss:.3f}\
        \tValidation Loss: {valid_loss:.3f} \tTrain Acc: {t_acc:.3f} \tValidation Acc: {v_acc:.3f}')
    
    sheet.cell(2+epoch-1, 1).value = epoch
    sheet.cell(2+epoch-1, 2).value = train_loss
    sheet.cell(2+epoch-1, 3).value = t_acc
    sheet.cell(2+epoch-1, 4).value = valid_loss
    sheet.cell(2+epoch-1, 5).value = v_acc
    sheet.cell(2+epoch-1, 6).value = time_elapsed
    
print("Training Finished")
