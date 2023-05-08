import openai

api_key = "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"
try:
    result = openai.ChatCompletion.create(
        api_key=api_key,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"},
        ],
        stop=["\n\n"],
        max_tokens=64,
        top_p=1,
    )
except Exception as e:
    print("***code API error***", api_key, str(e))
