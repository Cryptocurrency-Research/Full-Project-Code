# Import module
from tkinter import *

# Create object
root = Tk()

#Title
root.title("CRYPTO-CONVERTOR")
# Background
photo=PhotoImage(file="convert.png")
img=Label(image=photo)
img.grid(column=3,row=2)
# Adjust size
root.geometry("800x400")


bitcoinValue = 3452381.61
ethereum = 262487.54
binance = 33310.19


def convert(inrText):
    textD2 = dropdown2.get()
    print(inrText)
    if (textD2 == "BitCoin"):
        inrText = inrText / bitcoinValue
        label.config(text="BitCoin :  " + str(inrText))
    elif (textD2 == "Ethereum"):
        inrText = inrText / ethereum
        label.config(text="Ethereum :  " + str(inrText))
    else:
        inrText = inrText / binance
        label.config(text="Binance :  " + str(inrText))


# Change the label text
def show():
    inputValue = textBox.get("1.0", "end-1c")
    print(inputValue)
    textD1 = dropdown1.get()
    if (textD1 == "BitCoin"):
        valueCoin = (int(inputValue) * bitcoinValue)
        convert(valueCoin)

    if (textD1 == "Ethereum"):
        valueCoin = (int(inputValue) * ethereum)
        convert(valueCoin)

    if (textD1 == "Binance"):
        valueCoin = (int(inputValue) * binance)
        convert(valueCoin)

# Create Label
label1 = Label(root, text="CryptoCurrency  Research And Analytical Tool")
label1.grid(column=3, row=0)

# Create Label
label1 = Label(root, text="CryptoCurrency Conversion")
label1.grid(column=3, row=1)


# Dropdown menu options
options = ["BitCoin", "Ethereum", "Binance"]
# datatype of menu text
dropdown1 = StringVar()

# initial menu text
dropdown1.set("BitCoin")
# Create Dropdown menu
drop = OptionMenu(root, dropdown1, *options)
drop.grid(column=1, row=3)

#input Text
# TextBox Creation
textBox = Text(root, height=2, width=10)
textBox.grid(column=1, row=2)

# Create button, it will change label text
button = Button(root, text="Convert", command=show, bg="gold", activebackground='white').grid(column=3, row=3)

# Create Dropdown menu 2

dropdown2 = StringVar()
# initial menu text
dropdown2.set("BitCoin")
drop2 = OptionMenu(root, dropdown2, *options )
drop2.grid(column=5, row=3)

# Create Label
label = Label(root, text=" ")
label.grid(column=5, row=2)

# Execute tkinter
root.mainloop()
