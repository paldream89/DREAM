# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:54:41 2020

@author: xlm69
"""


import wx

app = wx.App()

frame = wx.Frame(None, -1, 'win.py')
frame.SetSize(0,0,200,50)

# Create open file dialog
openFileDialog = wx.FileDialog(frame, "Open", "", "", 
      "Bin files (*.bin)|*.bin", 
       wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST)

openFileDialog.ShowModal()
a = openFileDialog.GetPaths()
openFileDialog.Destroy()