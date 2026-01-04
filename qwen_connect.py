from openai import OpenAI
import requests
import json
import os
import httpx


def get_openai_client(key):
    # For OpenAI library v1.23.0+, create httpx client with explicit transport settings
    # This avoids proxy-related issues
    try:
        # Create httpx client that explicitly ignores system proxies
        http_client = httpx.Client(
            transport=httpx.HTTPTransport(retries=3),
            timeout=30.0
        )
        
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            http_client=http_client,
        )
    except Exception as e:
        # Fallback: try without custom http_client
        print(f"Warning: Could not create custom http_client ({e}), trying default initialization")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )
    
    return client

def get_response_from_openai(client, prompt):

    completion = client.chat.completions.create(
      extra_headers={
        "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
        "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
      },
      extra_body={},
      # model="qwen/qwq-32b:free",
      # model="qwen/qwen3-235b-a22b-07-25:free",
      # model="Qwen/Qwen3-235B-A22B-Instruct-2507"
      # model="qwen/qwen3-235b-a22b:free",
      model="qwen/qwen-2.5-vl-7b-instruct:free",
      messages=[
        {
          "role": "user",
          "content": prompt
        }
      ]
    )
    response = completion.choices[0].message.content
    print(response)
    return response


def get_response_direct(key, prompt):
    response = requests.post(
      url="https://openrouter.ai/api/v1/chat/completions",
      headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
        "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
      },
      data=json.dumps({
        # "model": "qwen/qwq-32b:free",
        # "model": "qwen/qwen3-235b-a22b-07-25:free",
        "model": "qwen/qwen-2.5-vl-7b-instruct:free",
        "messages": [
          {
            "role": "user",
            "content": prompt
          }
        ],
        
      })
    )      
    print(response)
    return response


def check_limitations(key):
    response = requests.get(
        url="https://openrouter.ai/api/v1/auth/key",
        headers={
            "Authorization": f"Bearer {key}",
        }
    )
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":

    check_limitations()
    exit(0)


    print('running direct connection example...')
    prompt_1 = "bob has 3 eggs, alice has 2 eggs. how many eggs do they have in total?" 
    response_1 = get_response_direct(prompt_1)

    prompt_2 = "bob gives 1 egg to alice. how many eggs does alice have now?"
    response_2 = get_response_direct(prompt_2)



    ### openai client example:
    # print('Running OpenAI client example...')
    # client = get_openai_client()
    # prompt_1 = "bob has 3 eggs, alice has 2 eggs. how many eggs do they have in total?" 
    # response_1 = get_response_from_openai(client, prompt_1)

    # print('Running OpenAI client example...')
    # client = get_openai_client()
    # prompt_2 = "bob gives 1 egg to alice. how many eggs does alice have now?"
    # response_2 = get_response_from_openai(client, prompt_2)



      