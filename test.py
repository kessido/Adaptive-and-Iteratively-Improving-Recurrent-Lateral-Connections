#!/usr/bin/env python
# coding: utf-8

# In[9]:


from helpers.data import get_dataloaders
from helpers.train import TrainingManager
from helpers.loss_accuracy import accuracy
from models.resnet import resnet20, resnet110
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


# In[10]:


trainloader, testloader = get_dataloaders('cifar10', 32)


# In[11]:


device = torch.device('cuda')


# In[12]:


model = resnet110()
model = model.to(device)
optimizer = Adam(model.parameters(), 2e-4)


# In[13]:


trial_name = "resnet110_adam_2e-4"


# In[14]:


tm = TrainingManager(trial_name, load=True, is_trial=True)


# In[15]:


tm.train(model, optimizer, 
         trainloader, testloader, 
         CrossEntropyLoss(), CrossEntropyLoss(), 
         accuracy, accuracy, device=device, no_iterations=100000)


# In[ ]:




