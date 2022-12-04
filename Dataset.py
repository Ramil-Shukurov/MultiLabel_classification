custom_transform = transforms.Compose([
                                       transforms.Resize((200,200)),
                                       transforms.ToTensor(),
                                       ])

path = os.path.join(os.getcwd(), "#4 Garbage_classification")
folders = list()
test_dataset = []
train_dataset = []
validation_dataset = []

for folder in os.listdir(path):  folders.append(folder)

for folder_idx in range(len(folders)):
    fpath = os.path.join(path, folders[folder_idx])
    pics = list()
    
    for pic in os.listdir(fpath): pics.append(pic)
    num1 = int(len(pics)*0.8)
    num2 = int(len(pics)*0.1)
    
    for img in pics:
        im = Image.open(os.path.join(fpath, img))
        im = custom_transform(im)
        if img in pics[:num1]: train_dataset.append([im, folder_idx])
        elif img in pics[num1:num1+num2]: validation_dataset.append([im, folder_idx])
        else: test_dataset.append([im, folder_idx])
        
test_loader = DataLoader(dataset = test_dataset, batch_size = 8, shuffle = False)
train_loader = DataLoader(dataset = train_dataset, batch_size = 24, shuffle = True)
validation_loader = DataLoader(dataset = validation_dataset, batch_size = 8, shuffle = True)
