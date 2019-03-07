<%@ page language="java" contentType="text/html; charset=UTF-8"	pageEncoding="UTF-8"%>
<%@ page import="c.Analyze"%>
<%
	request.setCharacterEncoding("UTF-8");
	String text = request.getParameter("input_text");
	c.Analyze ana = new c.Analyze(text);
	String result = ana.getVal();
	request.setAttribute("result", result);
	RequestDispatcher rd = request.getRequestDispatcher("index.jsp");
	rd.forward(request, response);
%>
