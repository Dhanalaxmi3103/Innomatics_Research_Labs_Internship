from flask import Flask, request
from datetime import datetime

app = Flask(__name__)

# ⏰ Time-based greeting
def greet_user():
    hour = datetime.now().hour
    if hour < 12:
        return "Good Morning ☀️"
    elif hour < 18:
        return "Good Afternoon 🌤️"
    else:
        return "Good Evening 🌙"

# 🔁 Reverse the name
def reverse_name(name):
    return name[::-1]

# 📏 Count vowels in the name
def count_vowels(name):
    vowels = "aeiouAEIOU"
    return sum(1 for char in name if char in vowels)

# 🎯 Palindrome check
def is_palindrome(name):
    clean_name = name.lower()
    return clean_name == clean_name[::-1]

@app.route("/")
def home():
    name = request.args.get("name")

    if not name:
        return "<h2>Please provide a name using ?name=YourName</h2>"

    return f"""
    <html>
        <body style="font-family:Arial; background:#eef2f3; padding:40px;">
            <h1>{greet_user()}, {name.upper()} 👋</h1>

            <p><b>Uppercase:</b> {name.upper()}</p>
            <p><b>Lowercase:</b> {name.lower()}</p>
            <p><b>Reversed:</b> {reverse_name(name)}</p>
            <p><b>Character Count:</b> {len(name)}</p>
            <p><b>Vowel Count:</b> {count_vowels(name)}</p>
            <p><b>Palindrome:</b> {"Yes ✅" if is_palindrome(name) else "No ❌"}</p>
        </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)

