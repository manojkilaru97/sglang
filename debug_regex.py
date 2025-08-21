#!/usr/bin/env python3
import re

text = '<｜tool▁calls▁begin｜>MATH_EVAL<｜tool▁sep｜>\n```json\n{\n  "expression": "sqrt(144)"\n}\n```\n<｜tool▁call▁end｜>'
pattern = r'<｜tool▁calls▁begin｜>(?P<name>[^\n<｜]+)<｜tool▁sep｜>\n```json\n(?P<args>[\s\S]*?)\n```(?:\n)?<｜tool▁call▁end｜>'

print("Text:", repr(text))
print("Pattern:", repr(pattern))

m = re.search(pattern, text, re.DOTALL)
if m:
    print('MATCH FOUND!')
    print('Name:', repr(m.group('name')))
    print('Args:', repr(m.group('args')))
else:
    print('NO MATCH')
    
# Let's also test if the bot token is found
bot_token = "<｜tool▁calls▁begin｜>"
print(f"Bot token found: {bot_token in text}")
