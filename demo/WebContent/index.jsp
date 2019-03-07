<%@ page language="java" contentType="text/html; charset=UTF-8"	pageEncoding="UTF-8"%>
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8xz">
<title>小蜗壳 - 细粒度情感分析平台</title>
<script type="text/javascript" src="jquery.js"></script>
<script type="text/javascript">
	function init() {
		var result = '<%=(String)request.getAttribute("result")%>'
		var text = '<%=(String)request.getParameter("input_text")%>'
		if(result != null && text != "输入中文" && result != "null"){
			$("#output").empty();
			$("#input").empty();
			$("#output").append(result.replace(/c/g, '\n'));
			$("#input").append(text);
		}
		time();
	}

	function time() {
		setTimeout("time()", 1000);
		var minutes = checkTime(new Date().getMinutes());
		var hours = checkTime(new Date().getHours());
		var sec = checkTime(new Date().getSeconds());
		document.getElementById("cc").innerHTML = "北京时间:" + hours + ":"
				+ minutes + ":" + sec;
	}

	function checkTime(obj) {
		if (obj < 10) {
			obj = "0" + obj;
		}
		return obj;
	}

	function reload() {
		location.href = "index.jsp";
	}
</script>
<style type="text/css">
h2 {
	text-align: center;
	margin: 50px auto;
}

#textarea {
	display: block;
	margin: 0 auto;
	overflow: hidden;
	width: 550px;
	font-size: 14px;
	height: 18px;
	line-height: 24px;
	padding: 2px;
}

textarea {
	outline: 0 none;
	border-color: rgba(82, 168, 236, 0.8);
	box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1), 0 0 8px
		rgba(82, 168, 236, 0.6);
}
</style>
</head>
<body onload="init()">
	<div
		style="text-align: center; padding-left: 30%; padding-right: 30%; margin-top: 20px; margin-left: auto; margin-right: auto;">
		<h1>情感分析</h1>
		<p id="cc"></p>
		<form action="handler.jsp" method="post">
			<textarea cols="50" rows="10" id="input" name="input_text"
				style="width: 100%; BORDER-color: #000;">输入中文</textarea>
			<input type="submit" value="开始分析" />
		</form>
		<textarea cols="50" rows="10" id="output" name="output_text"
			readonly="readonly"
			style="width: 100%; BORDER-width: 0px; height: 350px">在这里显示结果
		</textarea>
		<script> 
	        var text = document.getElementById("textarea");
	        autoTextarea(text);// 调用
        </script>
	</div>
</body>
</html>
