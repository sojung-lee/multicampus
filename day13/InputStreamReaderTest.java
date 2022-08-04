package day13;
/* InputStreamReader
 * - 2byte(문자) 기반 스트림
 * - 브릿지 스트림 
 * - 1byte로 읽어들인 데이터를 2byte로 조합해서 읽어들인다
 * - charset을 맞추는 기능이 있다.
 * */
import java.io.*;
public class InputStreamReaderTest {

	public static void main(String[] args) 
	throws IOException
	{
		String fname="C:\\Java\\Workspace\\Begin\\src\\day13\\FileInputStreamTest.java";//UTF-8
		//String fname="C:/myjava/readme.txt";//ANSI로 저장 ==> CP949==>EUC-KR
		
		File file=new File(fname);
		long fsize=file.length();//파일의 크기를 반환한다.
		System.out.println("파일의 크기: "+fsize+"bytes");
		
		//FileReader fr=new FileReader(file);//노드 연결
		FileInputStream fis=new FileInputStream(file);//노드 연결
		InputStreamReader fr=new InputStreamReader(fis,"UTF-8");
		
		OutputStreamWriter ow=new OutputStreamWriter(System.out,"UTF-8");
		
		int n=0;
		while((n=fr.read())!=-1) {
			ow.write(n);
			ow.flush();
			//System.out.write(n);
			//System.out.flush();
		}
		fr.close();
		

	}

}








