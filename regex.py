from flask import Flask, request
import re

app = Flask(__name__)

def highlight_matches(text, matches):
    highlighted = text
    offset = 0

    for i in matches:
        start, end = i.start() + offset, i.end() + offset
        highlighted = (
            highlighted[:start]
            + "<mark>"
            + highlighted[start:end]
            + "</mark>"
            + highlighted[end:]
        )
        offset += len("<mark></mark>")

    return highlighted

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    error = ""
    highlighted_text = ""
    test_string = ""
    pattern = ""

    if request.method == "POST":
        test_string = request.form["test_string"]
        pattern = request.form["regex"]

        try:
            regex = re.compile(pattern)
            matches = list(regex.finditer(test_string))

            for match in matches:
                results.append({
                    "match": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })

            highlighted_text = highlight_matches(test_string, matches)

        except re.error as e:
            error = str(e)

    return f"""
    <html>
    <head>
        <title>Regular Expression Matcher</title>
    </head>
    <body style="font-family:Arial; background:#f5f7fa; padding:40px;">
        <h2>🔍 Regular Expression Matcher(regex101 Core Clone)</h2>

        <form method="POST">
            <label><b>Test String</b></label><br>
            <textarea name="test_string" rows="5" cols="70">{test_string}</textarea><br><br>

            <label><b>Regex Pattern</b></label><br>
            <input type="text" name="regex" size="40" value="{pattern}"><br><br>

            <button type="submit">Submit</button>
        </form>

        <hr>

        {"<h3>Highlighted Matches</h3><p>" + highlighted_text + "</p>" if highlighted_text else ""}

        {"<h3>Match Details</h3>" if results else ""}
        <table border="1" cellpadding="8">
            <tr>
                <th>Match</th>
                <th>Start Index</th>
                <th>End Index</th>
            </tr>
            {''.join(f"<tr><td>{i['match']}</td><td>{i['start']}</td><td>{i['end']}</td></tr>" for i in results)}
        </table>

        {"<p><b>Total Matches:</b> " + str(len(results)) + "</p>" if results else ""}
        {"<p style='color:red;'><b>Error:</b> " + error + "</p>" if error else ""}
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)
