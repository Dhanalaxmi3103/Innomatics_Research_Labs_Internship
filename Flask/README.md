## Creative Flask App – Username Processor

A simple yet creative Flask web application that takes a username from the URL query parameter, converts it to UPPER CASE, and performs multiple fun text operations using custom Python functions.

This project demonstrates basic Flask routing, query parameter handling, and dynamic HTML responses.

##  Features

 🔠 Convert username to **UPPERCASE**
 🔡 Convert username to **lowercase**
 🔁 Reverse the username
 📏 Count total characters
 🔊 Count vowels in the name
 🎯 Check if the name is a palindrome
 ⏰ Display time-based greeting (Morning / Afternoon / Evening)



##  Technologies Used

 Python 3
 Flask
 HTML (rendered directly from Flask)



##  How to Run the Application

1️. Install Flask

```bash
pip install flask
```

2️. Run the Flask App

```bash
python app.py
```

3️. Open in Browser

```
http://127.0.0.1:5000/?name=yourname
```

### Example:

```
http://127.0.0.1:5000/?name=level
```


## 🖥️ Sample Output

```
Good Evening 🌙, LEVEL 👋
Uppercase: LEVEL
Lowercase: level
Reversed: level
Character Count: 5
Vowel Count: 2
Palindrome: Yes ✅
```

---

##  How It Works

* The app reads the `name` value from the URL using Flask’s `request.args`
* Custom Python functions process the input
* Results are dynamically displayed in the browser using HTML

---


