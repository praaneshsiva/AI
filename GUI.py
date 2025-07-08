import tkinter as tk
from tkinter import StringVar, IntVar, PhotoImage, Label, Entry, Checkbutton, Button, messagebox, filedialog

def check():
    username = user_name.get()
    pwd = password.get()
    if username == "admin" and pwd == "admin":
        messagebox.showinfo("Login Success", "Classify to will!")
        select_test_folder()
    else:
        messagebox.showerror("Login Failed", "Invalid Username or Password")

def clear():
    user_name.set("")
    password.set("")

def select_test_folder():
    folder_selected = filedialog.askdirectory(title="Select Test Resumes Folder")
    if folder_selected:
        messagebox.showinfo("Folder Selected", f"Selected folder: {folder_selected}")
        open_dashboard()

def open_dashboard():
    win.destroy()
    dashboard = tk.Tk()
    dashboard.title("Resume Classifier - Dashboard")
    dashboard.geometry("1000x600")
    tk.Label(dashboard, text="Welcome to the Resume Classifier Dashboard", font=("Times New Roman", 30), fg="#A48B48").pack(pady=20)
    dashboard.mainloop()

win = tk.Tk()
win.title("Resume classifier - LOGIN PAGE")
win.geometry("1000x500")

bg_image = PhotoImage(file="pic-1.png")
x = Label(win, image=bg_image)
x.place(x=0, y=0)
heading1 = Label(win, text='Resume Classifier', font=('TimesNewRoman', 40), fg="#A48B48", bg="#1F2C3F")
heading1.place(x=300, y=50)
heading = Label(win, text="Login", font='Verdana 25 bold', fg="#A48B48", bg="#C7C7A9")
heading.place(x=80, y=150)

username = Label(win, text="User Name:", font='Verdana 12 bold', fg="#537D58", bg="#C7C7A9")
username.place(x=80, y=220)
userpass = Label(win, text="Password:", font='Verdana 12 bold', fg="#537D58", bg="#C7C7A9")
userpass.place(x=80, y=260)
show_password_label = Label(win, text="Show password", font='Verdana 10 bold', fg="#537D58", bg="#C7C7A9")
show_password_label.place(x=480, y=260)

user_name = StringVar()
password = StringVar()
userentry = Entry(win, width=40, textvariable=user_name)
userentry.focus()
userentry.place(x=200, y=223)
passentry = Entry(win, width=40, show="*", textvariable=password)
passentry.place(x=200, y=260)

def mark():
    if var.get() == 1:
        passentry.configure(show="")
    else:
        passentry.configure(show="*")

var = IntVar()
bt = Checkbutton(win, command=mark, variable=var, offvalue=0, onvalue=1)
bt.place(x=450, y=260)
btn_login = Button(win, text="Login", font='Verdana 10 bold', fg="#BC6E25", bg="#C7C7A9", command=check)
btn_login.place(x=210, y=290)
btn_clear = Button(win, text="Clear", font='Verdana 10 bold', fg="#BC6E25", bg="#C7C7A9", command=clear)
btn_clear.place(x=260, y=290)

win.mainloop()
