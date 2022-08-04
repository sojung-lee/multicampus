package day10;
import java.util.*;
/* LinkedList : 인접 참조를 링크해서 체인처럼 관리하는 컬렉션
 * 				데이터를 저장한 후 수시로 데이터를 삽입하거나 삭제, 수정해야 할 경우 적합
 * ArrayList : 순차적으로 데이터를 저장할 때 적합함. 검색할 때 상대적으로 빠름
 * 
 * */

public class LinkedListTest {

	public static void main(String[] args) {
		List<String> list1=new ArrayList<>();
		List<String> list2=new LinkedList<>();
		//ArrayList 데이터 1만건을 저장 후 수행시간 측정
		
		long startTime = System.nanoTime();
		for(int i=0;i<100000;i++) {
			list1.add(0,"Hello "+i);
		}
		long endTime= System.nanoTime();
		
		long gapTime=endTime-startTime;
		System.out.println("***ArrayList 걸린 시간: "+gapTime+" ns");
		System.out.println("list1.size(): "+list1.size());
		System.out.println("********************************************");
		
		//for(String str:list1)
			//System.out.println(str);
		
		startTime = System.nanoTime();
		for(int i=0;i<100000;i++) {
			list2.add(0,"Hello "+i);
		}
		endTime =System.nanoTime();
		gapTime= endTime- startTime;
		System.out.println("LinkedList 걸린 시간: "+gapTime+"ns");
		/**
		 * 			순차적으로 추가, 삭제	|중간에 삽입, 삭제		| 검색
		 * ArrayList 	: 빠르다			| 느리다				| 빠르다
		 * LinkedList	: 느리다			| 빠르다				| 느리다
		 * */
		

	}

}
