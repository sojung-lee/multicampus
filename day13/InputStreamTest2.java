package day13;
import java.io.*;
//System.in : 키보드와 노드 연결
//System.out: 콘솔과 노드 연결
public class InputStreamTest2 {

	public static void main(String[] args)  
	throws IOException
	{
		int n=0;
		int cnt=0;
		System.out.println("입력하세요=>");
		//달걀(데이터)을 1개씩 이동 (1바이트씩 이동)
		while((n=System.in.read())!= -1) {//Ctrl+C or Ctrl+D을 입력하면 -1을 반환
			//System.out.println("n: "+((char)n));
			System.out.write(n);
			cnt++;
		}
		System.out.println(cnt+"bytes 읽음");
		
		System.in.close();
		System.out.close();

	}

}
