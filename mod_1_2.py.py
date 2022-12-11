from tkinter import *
root  = Tk()

# defining geometry
root.geometry("820x550")
root.resizable(0,0)
root.configure(bg='navy')
# root.minsize(650,550)

#  --------------------------------------------------------------------------------

bitcoinValue = 2342596
ethereum = 160959
binance = 22063


def convert(inrText):
    textD2 = dropdown2.get()
    print(inrText)
    if (textD2 == "BitCoin"):
        inrText = inrText / bitcoinValue
        label.config(text="You can buy :  " + str(inrText)+ " \nBitcoins")
    elif (textD2 == "Ethereum"):
        inrText = inrText / ethereum
        label.config(text="You can buy : " + str(inrText) + "\nEthereum")
    else:
        inrText = inrText / binance
        label.config(text="You can buy : " + str(inrText) + "\nBinance")


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

# --------------------------------------------------------------------------------

# Giving Lables
un = Label( root,text="* * *",font=("Rockwell Extra Bold",17),bg="navy",fg="white")
un.grid(row=0,column=0,pady=10)
un = Label( root,text="CryptoCurrency Conversion",font=("Rockwe ll Extra Bold",22),bg="navy",fg="white")
un.grid(row=0,column=1)
un = Label( root,text="* * *",font=("Rockwell Extra Bold",17),bg="navy",fg="white")
un.grid(row=0,column=2,pady=10)

text="Currently the data is stored offline, so values may vary in real time...!!!         "

text = (' '*30) + text + (' '*30)

marquee = Text(root, height=1, width=20)
marquee.place(x = 300, y = 500)

i = 0

def command(x, i):
    marquee.insert("1.1", x)
    if i == len(text):
        i = 0
    else:
        i = i+1
    root.after(100, lambda:command(text[i:i+20], i))

command(text[i:i+20], i)

# --------------------------------------------------------------------------------
options = ["BitCoin", "Ethereum", "Binance"]
# datatype of menu text
dropdown1 = StringVar()
# --------------------------------------------------------------------------------
# initial menu text
dropdown1.set("\nConvert from\n")  # set value to dropbox
# Create Dropdown menu
drop = OptionMenu(root, dropdown1, *options)
# drop.place(x =100, y = 280)
drop.grid(row=4,column=0 ,padx=25,pady=5)


# Create button, it will change label text
button = Button(root, text="Convert", command=show,font=("Stencil",17)).grid(column=1, row=4)

#input Text
# TextBox Creation
un = Label( root,text="Enter Value here",font=("Avenir Next LT Pro Light",15),bg="navy",fg="white")
un.grid(row=2,column=0,pady=20)
# un.place(x = 130, y = 100)
textBox = Text(root, height=2, width=10,font=(" ",14),pady=3, padx=3)
textBox.grid(column=1, row=2)

# Create Dropdown menu (Convert to)
dropdown2 = StringVar()
# initial menu text
dropdown2.set("\nConvert to\n")
drop2 = OptionMenu(root, dropdown2, *options)
drop2.grid(row = 4,column=2,padx=25,pady=55)
# drop.place(x =150, y = 50)

# Create Label
label = Label(root, text=" ")
label.grid(column=5, row=3)

# Execute tkinter
root.mainloop()