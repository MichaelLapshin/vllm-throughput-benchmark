import requests

WATTSUP_TOTAL = "wattsup_energy_joules_total"

def query_prometheus(metric: str, start_time_s: float, end_time_s: float, url: str = "http://localhost:9020/api/v1/query"):
    assert end_time_s >= start_time_s 
    duration_seconds = int(end_time_s - start_time_s)
    if duration_seconds == 0:
        return []
    query = f"{metric}[{duration_seconds}s] @ {end_time_s}"
    try:
        resp = requests.get(url, params={"query": query})
        resp.raise_for_status()
        data = resp.json()
        if data['status'] == 'success':
            return data['data']['result'][0]['values']
        else:
            print(f"Query Error: {data.get('error')}")
            return []
    except Exception as e:
        print(f"Error querying Prometheus: {e}")
        return []

def get_energy_joules(start_time_s: float, end_time_s: float) -> int:
    data = query_prometheus(WATTSUP_TOTAL, start_time_s, end_time_s)
    if data:
        return int(float(data[-1][1])) - int(float(data[0][1]))
    else:
        return 0
