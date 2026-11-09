"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
# PERSONA
Bạn là VinAssistant, trợ lý AI chính thức hỗ trợ khách hàng của Vingroup.
Bạn giao tiếp bằng tiếng Việt, lịch sự, chuyên nghiệp, thân thiện và rõ ràng.
Ưu tiên cung cấp thông tin chính xác, ngắn gọn và hữu ích.

# AVAILABLE TOOLS
Bạn có quyền sử dụng các công cụ sau:

1. search_product_catalog(category, max_price)
   - Tìm kiếm sản phẩm trong danh mục Vingroup.
   - category: xe_dien hoặc du_lich.
   - max_price: mức giá tối đa của sản phẩm.

2. submit_support_ticket(customer_name, issue_description, priority)
   - Tạo yêu cầu hỗ trợ khách hàng.
   - customer_name: tên khách hàng.
   - issue_description: nội dung cần hỗ trợ.
   - priority: low, medium hoặc high.

# CORE RULES
1. Không được bịa đặt giá cả, thông tin sản phẩm, tình trạng hoặc mã yêu cầu.
2. Khi khách hàng hỏi tìm sản phẩm, bắt buộc gọi search_product_catalog.
3. Khi khách hàng yêu cầu hỗ trợ hoặc báo cáo sự cố, bắt buộc gọi submit_support_ticket.
4. Nếu một yêu cầu chứa cả hai ý định, phải thực hiện cả hai công cụ độc lập.
5. Chỉ sử dụng dữ liệu thực tế trả về từ công cụ.
6. Nếu không tìm thấy dữ liệu, phải thông báo rõ ràng.
7. Không khẳng định một thao tác đã thành công nếu công cụ chưa xác nhận.

# OPERATIONAL BOUNDARIES
1. Chỉ hỗ trợ các sản phẩm, dịch vụ và hoạt động thuộc Vingroup.
2. Không trả lời các yêu cầu ngoài phạm vi Vingroup.
3. Không tiết lộ dữ liệu nội bộ, thông tin cá nhân hoặc dữ liệu khách hàng khác.
4. Không tự ý thay đổi dữ liệu hoặc thực hiện hành động ngoài các công cụ được cấp.
5. Nếu yêu cầu không rõ ràng, hãy hỏi khách hàng để làm rõ.

# OUTPUT CONTRACT
Quá trình xử lý tuân theo cấu trúc ReAct:

Thought:
Phân tích yêu cầu và xác định ý định của khách hàng.

Action:
Xác định công cụ cần gọi và các tham số tương ứng.

Observation:
Ghi nhận kết quả thực tế trả về từ công cụ.

Final Answer:
Tổng hợp kết quả thành câu trả lời tiếng Việt rõ ràng,
không tiết lộ suy luận nội bộ và không đưa ra thông tin chưa được xác minh.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }



# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering và Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max(1, max_iterations)
        self.trace: List[Dict[str, Any]] = []

    def _detect_intent(self, user_input: str) -> Dict[str, Any]:
        """Nhận diện các ý định của khách hàng."""

        text = user_input.lower()

        catalog_search_keywords = [
    "tìm sản phẩm",
    "tìm kiếm sản phẩm",
    "muốn xem",
    "xem sản phẩm",
    "tìm mua",
    "muốn mua",
    "có xe",
    "xe điện nào",
    "danh sách sản phẩm",
    "giá dưới",
    "giá khoảng",
    "sản phẩm nào"
]

        ticket_keywords = [
    "khiếu nại",
    "báo lỗi",
    "báo sự cố",
    "bị lỗi",
    "gặp sự cố",
    "không hoạt động",
    "bị hỏng",
    "gặp vấn đề",
    "phản ánh",
    "cần xử lý",
    "cần sửa",
    "yêu cầu hỗ trợ",
    "đăng ký hỗ trợ",
    "tôi muốn được hỗ trợ",
    "cần hỗ trợ"
]

        needs_catalog = any(
            keyword in text for keyword in catalog_search_keywords
        )

        needs_ticket = any(
            keyword in text for keyword in ticket_keywords
        )

        # Xác định danh mục sản phẩm
        if any(keyword in text for keyword in [
            "du lịch", "khách sạn", "resort",
            "vinpearl", "nghỉ dưỡng"
        ]):
            category = "du_lich"
        elif any(keyword in text for keyword in [
            "xe", "vinfast", "ô tô", "oto",
            "xe điện", "xe máy điện"
        ]):
            category = "xe_dien"
        else:
            category = None

        # Xác định mức giá tối đa
        max_price = 999999999999

        price_match = re.search(
            r"(?:dưới|thấp hơn|tối đa|không quá|nhỏ hơn)\s*"
            r"(\d+(?:[.,]\d+)?)\s*(tỷ|tỉ|triệu|tr)"
            , text
        )

        if price_match:
            value = float(price_match.group(1).replace(",", "."))

            unit = price_match.group(2)

            if unit in ["tỷ", "tỉ"]:
                max_price = int(value * 1_000_000_000)
            else:
                max_price = int(value * 1_000_000)

        # Xác định tên khách hàng nếu có
        name_match = re.search(
            r"(?:tôi tên(?: là)?|tên tôi là|tên là)\s+([^,.;]+)",
            user_input,
            re.IGNORECASE
        )

        customer_name = (
            name_match.group(1).strip()
            if name_match else "Khách hàng"
        )

        return {
            "catalog": needs_catalog and category is not None,
            "ticket": needs_ticket,
            "category": category,
            "max_price": max_price,
            "customer_name": customer_name
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""

        self.trace = []

        self.trace.append({
            "step": "init",
            "user_input": user_input
        })

        # TODO 3: Intent Detection
        intent = self._detect_intent(user_input)

        self.trace.append({
            "step": "intent_detection",
            "intent": intent
        })

        # Xây dựng danh sách các hành động cần thực hiện.
        actions = []

        if intent["catalog"]:
            actions.append({
                "tool": "search_product_catalog",
                "arguments": {
                    "category": intent["category"],
                    "max_price": intent["max_price"]
                }
            })

        if intent["ticket"]:
            priority = (
                "high"
                if any(keyword in user_input.lower() for keyword in [
                "nghiêm trọng", "khẩn cấp", "gấp", "nguy hiểm"
                ])
                else "medium"
            )

            actions.append({
                "tool": "submit_support_ticket",
                "arguments": {
                    "customer_name": intent["customer_name"],
                    "issue_description": user_input,
                    "priority": priority
                }
            })

        # TODO 4: Agent Loop
        observations = []
        iteration = 0
        action_index = 0

        while (
            action_index < len(actions)
            and iteration < self.max_iterations
        ):
            iteration += 1

            action = actions[action_index]
            tool_name = action["tool"]
            arguments = action["arguments"]

            self.trace.append({
                "step": "action",
                "iteration": iteration,
                "tool": tool_name,
                "arguments": arguments
            })

            try:
                tool_function = TOOL_MAP[tool_name]
                result = tool_function(**arguments)

                observation = {
                    "tool": tool_name,
                    "result": result,
                    "status": "success"
                }

            except Exception as error:
                observation = {
                    "tool": tool_name,
                    "result": str(error),
                    "status": "error"
                }

            observations.append(observation)

            self.trace.append({
                "step": "observation",
                "iteration": iteration,
                **observation
            })

            action_index += 1

        # Tổng hợp kết quả
        if action_index < len(actions):
            status = "max_iterations_reached"
        else:
            status = "completed"

        # Tạo câu trả lời cuối cùng
        answer_parts = []

        if not intent["catalog"] and not intent["ticket"]:
            iteration = 1
            if "bảo hành" in user_input.lower():
                answer_parts.append(
                    "Chính sách bảo hành pin xe điện VinFast phụ thuộc "
                    "vào dòng xe và điều kiện bảo hành áp dụng. "
                    "Vui lòng kiểm tra chính sách chính thức của VinFast "
                    "để xác định thời hạn chính xác."
                )
            else:
                answer_parts.append(
                    "VinAssistant sẵn sàng hỗ trợ các thông tin "
                    "về sản phẩm và dịch vụ thuộc Vingroup. "
                    "Vui lòng cung cấp thêm nội dung cần hỗ trợ."
                )

        for observation in observations:

            tool_name = observation["tool"]
            result = observation["result"]

            if observation["status"] == "error":
                answer_parts.append(
                    f"Không thể thực hiện {tool_name}: {result}"
                )
                continue

            if tool_name == "search_product_catalog":

                if isinstance(result, list) and result:
                    valid_products = [
                        item for item in result
                        if isinstance(item, dict)
                        and "error" not in item
                    ]

                    if valid_products:
                        answer_parts.append(
                            "Các sản phẩm phù hợp:\n"
                            + "\n".join(
                                f"- {item.get('name', 'N/A')}: "
                                f"{item.get('price_vnd', 'N/A'):,} VND"
                                if isinstance(item.get("price_vnd"), (int, float))
                                else f"- {item.get('name', 'N/A')}: "
                                     f"{item.get('price_vnd', 'Chưa có thông tin giá')}"
                                for item in valid_products
                            )
                        )
                    else:
                        answer_parts.append(
                            "Không tìm thấy sản phẩm phù hợp "
                            "trong danh mục."
                        )
                else:
                    answer_parts.append(
                        "Không tìm thấy sản phẩm phù hợp "
                        "với điều kiện tìm kiếm."
                    )

            elif tool_name == "submit_support_ticket":

                if isinstance(result, dict):
                    ticket_id = result.get("ticket_id", "Chưa xác định")
                    ticket_status = result.get("status", "Chưa xác định")
                    customer_name = observation.get("result", {}).get(
                        "customer_name", "Khách hàng"
                    )

                    answer_parts.append(
                        f"Yêu cầu hỗ trợ của {intent['customer_name']} đã được ghi nhận.\n"
                        f"Mã yêu cầu: {ticket_id}\n"
                        f"Trạng thái: {ticket_status}"
                    )
                else:
                    answer_parts.append(
                        "Đã gửi yêu cầu hỗ trợ."
                    )

        if status == "max_iterations_reached":
            answer_parts.append(
                "Quá trình xử lý đã đạt giới hạn số vòng lặp. "
                "Một số yêu cầu có thể chưa được thực hiện."
            )

        final_answer = "\n\n".join(answer_parts)

        self.trace.append({
            "step": "final_answer",
            "answer": final_answer,
            "status": status
        })

        return {
            "answer": final_answer,
            "trace": self.trace,
            "iterations": iteration,
            "status": status
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)

    print("\nResult:", result["answer"])
    print("Iterations:", result["iterations"])
    print("Status:", result["status"])

    print(
        "\nTrace Log:",
        json.dumps(agent.trace, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()

