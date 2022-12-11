from tkinter import *
root  = Tk()
root.configure(bg='navy')
root.geometry("800x550")
l1 = Label(root,text="\n\nThis Program converts the current value of cryptocurrency",font=("Arial Black",17),bg="navy",fg="white")
l1.pack()
l1 = Label(root,text="To",font=("Arial Black",16),bg="navy",fg="white")
l1.pack()
l1 = Label(root,text=" Indian Ruprees (INR)",font=("Arial Black",17),bg="navy",fg="white")
l1.pack()

# -------- Function

def forexit():
    exit()

btn1 = Button(text="Exit",command = forexit)
btn1.place(x =230, y = 250, width=90,height=50)

def browse_button():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global folder_path
    filename = filedialog.askdirectory()
    folder_path.set('mod_1_2.py')
    print('mod_1_2.py')
#btn2 = Button(text="Continue",command=)
# btn2.place(x =530, y = 250, width=90,height=50)

root.mainloop()