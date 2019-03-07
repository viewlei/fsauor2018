import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;

/**
 * 
 * 路径还需要改下
 * @author Erbenner
 *
 */
public class Analyze {
    public Process process;
    public String cmd[];
    public String result;
    public String text;
    public static String error = "通信出错/服务器未响应！";

    public Analyze(String text) {
        this.text = text;
        StringBuilder sb = new StringBuilder();
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < 20; i++) {
            String tmp = getSingalResult(12000 + i);
            if (tmp.equals(error)) {
                continue;
            } else {
                sb.append(tmp + 'c');
            }
        }
        long endTime = System.currentTimeMillis();
        Long dt = (long) ((endTime - startTime));
        sb.append("本次计算耗时：" + dt.toString() + " msc");
        result = sb.toString();
    }

    public String getVal() {
        return this.result;
    }

    public String getSingalResult(int port) {
        try {
            Socket socket = new Socket("localhost", port);
            // 获取输出流，向服务器端发送信息
            OutputStream os = socket.getOutputStream();// 字节输出流
            PrintWriter pw = new PrintWriter(os);// 将输出流包装为打印流
            pw.write(text);
            pw.flush();
            socket.shutdownOutput();// 关闭输出流

            InputStream is = socket.getInputStream();
            BufferedReader in = new BufferedReader(new InputStreamReader(is));
            String info = in.readLine();
            is.close();
            in.close();
            socket.close();
            return info;
        } catch (UnknownHostException e) {
            e.printStackTrace();
            return error;
        } catch (IOException e) {
            e.printStackTrace();
            return error;
        }
    }

    public static void main(String[] args) {
    }
}
