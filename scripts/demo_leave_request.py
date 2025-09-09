# scripts/demo_leave_request.py
from tools.leave_request import create_leave_request, list_leave_requests, get_leave_request

def main():
    print("Creating a sample leave request…")
    req = create_leave_request(
        user="alice",
        start_date="2025-09-18",
        end_date="2025-09-20",
        reason="travel"
    )

    # bad date order
    # req = create_leave_request("alice", "2025-09-20", "2025-09-18", "oops")
    
    print("Created:", req)

    print("\nListing requests for alice:")
    print(list_leave_requests("alice"))

    print("\nFetching by id:")
    print(get_leave_request(req["id"]))

if __name__ == "__main__":
    main()
